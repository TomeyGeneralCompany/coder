# Required: ``LLMProfileStore.list()`` shadows the builtin in the class body,
# so annotations like ``list[dict[str, Any]]`` would fail without deferral.
from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from filelock import FileLock, Timeout

from openhands.sdk.errors import Invalid, LockTimeout
from openhands.sdk.llm.utils.openhands_provider import (
    canonicalize_openhands_llm_payload,
)
from openhands.sdk.logger import get_logger
from openhands.sdk.utils.cipher import FERNET_TOKEN_PREFIX
from openhands.sdk.utils.path import get_user_persistence_dir
from openhands.sdk.utils.pydantic_secrets import (
    REDACTED_SECRET_VALUE,
    SecretDecryptionError,
    cipher_context,
    client_context,
)


if TYPE_CHECKING:
    from openhands.sdk.llm.llm import LLM
    from openhands.sdk.llm.provider_connection_store import ProviderConnectionStore
    from openhands.sdk.utils.cipher import Cipher

_DEFAULT_PROFILE_DIR: Final[Path] = get_user_persistence_dir() / "profiles"
_LOCK_TIMEOUT_SECONDS: Final[float] = 30.0

# Profile names: 1-64 chars, must start with alphanumeric, then alphanumerics
# or '.', '_', '-'. Blocks empty names, path separators, leading dots
# (hidden files / path traversal), and shell-special characters.
PROFILE_NAME_PATTERN: Final[str] = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
PROFILE_NAME_REGEX: Final[re.Pattern[str]] = re.compile(PROFILE_NAME_PATTERN)


def canonical_profile_name(name: str) -> str:
    """Return the store key ``name`` addresses: ``foo.json`` is profile ``foo``."""
    return name.removesuffix(".json")


logger = get_logger(__name__)


class ProfileLimitExceeded(Exception):
    """Raised when saving would exceed the configured profile limit."""


# Historical cloud credential profiles are retained on disk until explicitly
# migrated or deleted; they are never applied to native local inference.
_LEGACY_INFERENCE_FIELDS = frozenset({
    "provider_connection_id", "api_key", "auth_type", "fallback_strategy",
    "openai_api_key", "provider", "litellm_extra_body", "api_mode",
})


def _check_native_profile(data: dict[str, Any]) -> None:
    found = sorted(_LEGACY_INFERENCE_FIELDS.intersection(data))
    if found:
        raise Invalid(
            "Legacy hosted-provider profile cannot be activated by local Coder: "
            + ", ".join(found)
            + ". Existing profile data was not modified."
        )
    if data.get("model", "qwen3:4b") != "qwen3:4b":
        raise Invalid(
            f"Only locally installed qwen3:4b is supported; found {data.get('model')!r}."
        )


@runtime_checkable
class LLMProfileLoader(Protocol):
    """Minimal load-only contract consumed by ``openhands.sdk.launch.resolve``.

    The resolver reads an LLM profile by name via ``load`` only, so an alternate
    backend — e.g. a cloud adapter over the ``org.llm_profiles`` column — need
    implement just this, not the full file-backed :class:`LLMProfileStore`.
    """

    def load(self, name: str, *, cipher: Cipher | None = ...) -> LLM: ...


@runtime_checkable
class LLMProfileMutator(Protocol):
    """Delete/rename contract used by the cross-store FK helpers.

    ``profile_refs.delete_llm_profile`` / ``rename_llm_profile`` touch the LLM
    store only through ``delete`` / ``rename``, so a cloud adapter over
    ``org.llm_profiles`` can drive the same guarded FK lifecycle without being a
    full :class:`LLMProfileStore`.
    """

    def delete(self, name: str) -> None: ...

    def rename(self, old_name: str, new_name: str) -> None: ...


class LLMProfileStore:
    """Standalone utility for persisting LLM configurations."""

    def __init__(
        self,
        base_dir: Path | str | None = None,
        *,
        provider_store: ProviderConnectionStore | None = None,
    ) -> None:
        """Initialize the profile store.

        Args:
            base_dir: Path to the directory where the profiles are stored.
                If `None` is provided, the default directory is used, i.e.,
                `~/.openhands/profiles`.
            provider_store: Store of shared provider connections used to
                resolve a profile's ``provider_connection_id`` at load time.
                When `None` (the default), a :class:`ProviderConnectionStore`
                is created in a ``provider-connections`` directory *sibling to*
                ``base_dir`` so that every ``LLMProfileStore`` — including
                custom-directory and bare standalone SDK instances — resolves
                connections from the same location it reads profiles from. Pass
                an explicit store to use an unrelated directory (e.g. the
                agent-server's config-scoped directory) or a test double.
        """
        self.base_dir = Path(base_dir) if base_dir is not None else _DEFAULT_PROFILE_DIR
        # ensure directory existence
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._file_lock = FileLock(self.base_dir / ".profiles.lock")
        # Retained constructor parameter for old caller compatibility only.
        # Shared hosted-provider credentials are not loaded or created.
        self._provider_store = None

    @contextmanager
    def _acquire_lock(self, timeout: float = _LOCK_TIMEOUT_SECONDS) -> Iterator[None]:
        """Acquire file lock for safe concurrent access.

        Args:
            timeout: Maximum time to wait for lock acquisition in seconds.

        Raises:
            TimeoutError: If the lock cannot be acquired within the timeout.
        """
        try:
            with self._file_lock.acquire(timeout=timeout):
                yield
        except Timeout:
            logger.error(f"[Profile Store] Failed to acquire lock within {timeout}s")
            raise LockTimeout(
                "Profile store is busy. Please retry.",
                message=f"Profile store lock acquisition timed out after {timeout}s",
            )

    def list(self) -> list[str]:
        """Returns a list of all profiles stored.

        Returns:
            List of profile filenames (e.g., ["default.json", "gpt4.json"]).
        """
        with self._acquire_lock():
            return [p.name for p in self.base_dir.glob("*.json")]

    def _get_profile_path(self, name: str) -> Path:
        """Get the full path for a profile name.

        Args:
            name: Profile name (must match ``PROFILE_NAME_PATTERN``).

        Raises:
            ValueError: If name does not match the allowed pattern.
        """
        clean_name = canonical_profile_name(name)
        if not PROFILE_NAME_REGEX.match(clean_name):
            raise Invalid(
                f"Invalid profile name: {name!r}. "
                "Profile names must be 1-64 characters, start with a letter "
                "or digit, and contain only letters, digits, '.', '_', or '-'."
            )
        return self.base_dir / f"{clean_name}.json"

    def save(
        self,
        name: str,
        llm: LLM,
        include_secrets: bool = False,
        *,
        cipher: Cipher | None = None,
        max_profiles: int | None = None,
    ) -> None:
        """Save a profile to the profile directory.

        Overwrites an existing profile of the same name. When ``max_profiles``
        is set, raises ``ProfileLimitExceeded`` if creating a *new* profile
        would exceed the limit. The check happens under the same lock as the
        save, so it is race-free against other ``save`` calls in this process.

        Args:
            name: Name of the profile to save.
            llm: LLM instance to save
            include_secrets: Whether to include the profile secrets. Defaults to False.
            cipher: Optional cipher for at-rest encryption of secrets.
                When provided, secrets are encrypted before writing to disk.
            max_profiles: Optional cap on the number of profiles.

        Raises:
            ProfileLimitExceeded: If ``max_profiles`` would be exceeded.
            SecretDecryptionError: If ``cipher`` is given and the existing
                profile holds a secret it cannot decrypt (a key change), so
                overwriting it would lose that secret.
            TimeoutError: If the lock cannot be acquired.
        """
        profile_path = self._get_profile_path(name)

        with self._acquire_lock():
            if max_profiles is not None and not profile_path.exists():
                # Only count files visible via list_summaries (valid names),
                # so stray invalid files don't consume slots.
                count = sum(
                    1
                    for p in self.base_dir.glob("*.json")
                    if PROFILE_NAME_REGEX.match(p.stem)
                )
                if count >= max_profiles:
                    raise ProfileLimitExceeded(
                        f"Profile limit reached ({max_profiles})."
                    )

            if profile_path.exists():
                if cipher is not None:
                    _raise_if_undecryptable(profile_path, cipher)
                logger.info(
                    f"[Profile Store] Profile `{name}` already exists. Overwriting."
                )

            # Native Qwen has no API keys or hosted provider secret fields.
            # Preserve the saved schema and atomic replace, not legacy secrets.
            _check_native_profile(llm.model_dump(mode="json"))
            profile_json = json.dumps(llm.to_persisted(), indent=2)
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.base_dir, suffix=".tmp", delete=False
            ) as tmp:
                tmp.write(profile_json)
                tmp_path = Path(tmp.name)

            try:
                Path.replace(tmp_path, profile_path)
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise
            logger.info(f"[Profile Store] Saved profile `{name}` at {profile_path}")

    def load(
        self,
        name: str,
        *,
        cipher: Cipher | None = None,
        resolve_provider: bool = True,
        strict: bool = False,
    ) -> LLM:
        """Load an LLM instance from the given profile name.

        Args:
            name: Name of the profile to load.
            cipher: Optional cipher for decrypting secrets stored at rest.
                When provided, encrypted secrets are decrypted during load.
            resolve_provider: When True (default) and the profile references a
                provider connection, its shared ``api_key`` / ``base_url`` are
                applied to the returned LLM (read-at-use). Set False to inspect
                the profile as stored without touching the provider store — used
                by display paths so a dangling reference never fails a read.
            strict: A secret ``cipher`` cannot decrypt loads as ``None`` by
                default. With ``strict=True`` (paths that write the loaded LLM
                elsewhere) the load raises ``SecretDecryptionError`` instead.

        Returns:
            An LLM instance constructed from the profile configuration, with
            provider-connection credentials applied when ``resolve_provider``.

        Raises:
            FileNotFoundError: If the profile name does not exist.
            ValueError: If the profile file is corrupted or invalid.
            SecretDecryptionError: With ``strict``, if a stored secret cannot
                be decrypted with ``cipher``.
            ProviderConnectionNotFound: If the profile references a provider
                connection that no longer exists and carries no inline key.
            TimeoutError: If the lock cannot be acquired.
        """
        profile_path = self._get_profile_path(name)

        with self._acquire_lock():
            if not profile_path.exists():
                existing = [p.name for p in self.base_dir.glob("*.json")]
                raise FileNotFoundError(
                    f"Profile `{name}` not found. "
                    f"Available profiles: {', '.join(existing) or 'none'}"
                )

            try:
                from openhands.sdk.llm.llm import LLM

                data = json.loads(profile_path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise Invalid("Profile must be a JSON object")
                _check_native_profile(data)
                llm_instance = LLM.from_persisted(data)
            except SecretDecryptionError:
                raise
            except Exception as e:
                # Re-raise as ValueError for clearer error handling
                raise Invalid(f"Failed to load profile `{name}`: {e}") from e

            logger.info(f"[Profile Store] Loaded profile `{name}` from {profile_path}")

        # resolve_provider/strict are retained as call-compatible options.
        # Neither ever activates an external inference provider.
        return llm_instance

    def resolve_provider_connection(
        self, llm: LLM, *, cipher: Cipher | None = None
    ) -> LLM:
        """Native Qwen requires no credential resolution."""
        _check_native_profile(llm.model_dump(mode="json"))
        return llm

    def delete(self, name: str) -> None:
        """Delete an existing profile.

        If the profile is not present in the profile directory, it does nothing.

        Args:
            name: Name of the profile to delete.

        Raises:
            TimeoutError: If the lock cannot be acquired.
        """
        profile_path = self._get_profile_path(name)

        with self._acquire_lock():
            if not profile_path.exists():
                logger.info(f"[Profile Store] Profile `{name}` not found. Skipping.")
                return

            profile_path.unlink()
            logger.info(f"[Profile Store] Deleted profile `{name}`")

    def rename(self, old_name: str, new_name: str) -> None:
        """Atomically rename a profile.

        Raises FileNotFoundError if ``old_name`` is missing, FileExistsError
        if ``new_name`` is taken. When the names resolve to the same path,
        the call is a no-op but still verifies the profile exists.
        """
        old_path = self._get_profile_path(old_name)
        new_path = self._get_profile_path(new_name)

        with self._acquire_lock():
            if not old_path.exists():
                raise FileNotFoundError(f"Profile `{old_name}` not found")
            if old_path == new_path:
                return
            if new_path.exists():
                raise FileExistsError(f"Profile `{new_name}` already exists")
            old_path.rename(new_path)
            logger.info(f"[Profile Store] Renamed profile `{old_name}` to `{new_name}`")

    def list_summaries(self) -> list[dict[str, Any]]:
        """List profile metadata without instantiating LLM objects.

        Reads JSON directly to avoid ``LLM._set_env_side_effects`` mutating
        ``os.environ``. Files with invalid names, corrupted JSON, or non-dict
        top-level values are skipped with a warning.

        A linked provider connection's key presence is read without a cipher:
        this method only reports whether a key is set, never its plaintext, so
        an encrypted-at-rest key still reads as present without decryption.
        """
        summaries: list[dict[str, Any]] = []
        with self._acquire_lock():
            for path in sorted(self.base_dir.glob("*.json")):
                name = path.stem
                if not PROFILE_NAME_REGEX.match(name):
                    logger.warning(
                        f"[Profile Store] Skipping profile with invalid name {name!r}"
                    )
                    continue
                try:
                    data = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError) as e:
                    logger.warning(
                        f"[Profile Store] Skipping corrupted profile {name!r}: {e}"
                    )
                    continue
                if not isinstance(data, dict):
                    logger.warning(
                        f"[Profile Store] Skipping non-dict profile {name!r}"
                    )
                    continue
                # Retain imported legacy profiles in the listing so owners can
                # identify and remove them; never expose credential values.
                legacy_fields = bool(_LEGACY_INFERENCE_FIELDS.intersection(data))
                summaries.append({
                    "name": name,
                    "model": data.get("model"),
                    "base_url": data.get("base_url"),
                    "provider_connection_id": data.get("provider_connection_id"),
                    "provider_connection_broken": legacy_fields,
                    "api_key_set": False,
                })
        return summaries
