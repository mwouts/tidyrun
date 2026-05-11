# Key Design Decisions in TidyRun Serialization

## 1. TOML-Based Key Encoding (No Prefix, No URL Encoding)

**Decision:** Key encoding uses `toml.dumps({_KEY_NAME: key})` and extracts the scalar value directly, without prefixes or URL escaping.

**Rationale:**
- TOML is a formal specification, ensuring type-safe round-tripping (int stays int, datetime stays datetime)
- Removing prefixes and URL encoding keeps filenames human-readable
- Validation happens post-extraction to reject problematic names (path separators, reserved prefixes)

**Trade-off:** Cannot distinguish between `42` (int) and `"42"` (string) in filenames, but that's acceptable since keys are typed during decode.

**Alternative Considered:** Storing type markers (e.g., `i:42` for int, `s:42` for string) — too verbose and harder to debug.

---

## 2. Metadata Sidecars Over Inferred Formats

**Decision:** Each serialized value gets a `.tidyrun` metadata sidecar recording encoding format, version, and suffix.

**Rationale:**
- Explicit metadata avoids ambiguity (e.g., `.json` could be from JSON encoder or pickle with `.json` name)
- Versioning enables future format migration without breaking old data
- Metadata is human-readable (TOML format)
- Allows deserialize to work extension-free (caller uses metadata to find payload)

**Trade-off:** Adds one extra small file per output, but worth the clarity and future-proofing.

**Alternative Considered:** Store metadata inside the file (e.g., Parquet metadata tags) — less portable, harder to inspect manually.

---

## 3. LazyDict Returns Instead of Eager Materialization

**Decision:** `deserialize()` returns `LazyDict` for dict-folder encodings instead of eagerly loading all nested values.

**Rationale:**
- DAG workflows often produce deeply nested outputs; eager loading wastes memory on unused branches
- Lazy evaluation aligns with functional data pipeline philosophy (only compute when needed)
- Each access reloads from storage, which keeps memory usage bounded
- Natural fit with `concat()` for analytics workflows

**Trade-off:** User code must expect `LazyDict` not plain `dict`; mitigated by `to_dict()` for explicit materialization.

**Alternative Considered:** Mixed eager/lazy based on heuristic (size, depth) — too complex and unpredictable.

---

## 4. GoToNextEncoderException for Fallback Signaling

**Decision:** Instead of hardcoded `try/except` for specific failure types, encoders raise `GoToNextEncoderException` to signal "skip me and try the next encoder."

**Rationale:**
- Decouples fallback logic from specific error types
- Extensible: custom encoders can signal fallback without modifying core logic
- Clearer intent: exception name says "go to next" not "I failed"
- Consistent with encoder pipeline philosophy

**Trade-off:** New exception type adds to public API, but semantics are clear to users.

**Alternative Considered:** Return sentinel value or use callbacks — less Pythonic and harder to trace through nested calls.

---

## 5. Generic Location Type (str | PathLike) with Explicit Remote Rejection

**Decision:** Accept `Location = str | PathLike[str]` and add optional `s3://` support via a separate backend.

**Rationale:**
- Future-proofs for fsspec integration without requiring it now
- S3 support is implemented without making boto3 a hard dependency
- Serialization and deserialization both stage through a temporary local tree, preserving the existing file-based format logic

**Trade-off:** S3 support depends on an optional dependency and temporary local staging.

**Alternative Considered:** Eagerly integrate fsspec now — increases dependency surface; better to keep the first remote backend focused.

---

## 6. Encoder Pipeline with Predicates

**Decision:** Encoders use predicates (functions returning bool) rather than exception-based dispatch.

**Rationale:**
- Predicates are composable and easy to test independently
- Order is explicit and easy to customize
- No exception overhead in the selection phase
- Mirrors established patterns (e.g., Pandas dtype resolution)

**Trade-off:** Predicates can be expensive (e.g., `is_json_serializable` tries `json.dumps`); users can optimize with custom pipeline.

**Alternative Considered:** Multiple dispatch or type registry — more complex; predicates are simpler for extension.

---

## 7. HDF5 with Fixed Key `"data"` Instead of DataFrame Column Name

**Decision:** HDF5 serialization stores data under key `"data"`, not the Series/DataFrame's name attribute.

**Rationale:**
- Consistent for both DataFrame and Series (both store at `"data"`)
- Series name is preserved in the pandas object itself
- Simpler deserialization (no need to parse or recover name from filename)

**Trade-off:** HDF5 files are less self-documenting than if column names were preserved.

**Alternative Considered:** Use Series name as key — breaks when Series is unnamed or when names collide.

---

## 8. Concat Without Recursive Transform at Intermediate Levels

**Decision:** `LazyDict.concat()` applies `transform` only to leaf values (DataFrames/Series), not to intermediate LazyDicts.

**Rationale:**
- Intermediate dicts (non-leaves) aren't concatenated; they're containers
- Transforming leaves is the intended use case (add metadata, normalize schema)
- Simpler semantics for users

**Trade-off:** If a user wants to transform intermediate structures, they must materialize and re-organize manually.

**Alternative Considered:** Recursive transform — too many edge cases and hard to specify.

---

## 9. Double Extensions Allowed (No `.json` Suffix Guard)

**Decision:** Removed checks that rejected serialize targets or deserialize sources with extensions.

**Rationale:**
- Adds flexibility (user can serialize to `output.v1` if desired)
- Metadata stores true suffix, so double extension doesn't cause confusion
- Simpler code; less surprising behavior
- Users already skilled enough to manage names

**Trade-off:** Less guidance for new users; mitigated by documentation and examples.

**Alternative Considered:** Strict no-extension policy — too rigid.

---

## 10. Public API via Serialization Package

**Decision:** Split implementation into `serialization/` subpackage and expose it directly.

**Rationale:**
- Public API now matches the implementation package name
- Clear internal module structure for future development

**Trade-off:** Slightly longer import path for users.

**Alternative Considered:** Keep a separate compatibility facade — no longer needed.

---

## 11. Metadata Version Field for Future Evolution

**Decision:** Added `version = 1` to metadata sidecars; deserialization validates it.

**Rationale:**
- Enables format evolution (v2, v3) without breaking v1 readers
- Future format changes can be conditional on version
- Guidance for deprecation strategy (support v1 indefinitely, warn on old versions)

**Trade-off:** Requires validation logic; minimal cost, high future value.

**Alternative Considered:** Implicit versioning (based on file absence/presence) — fragile; explicit is clearer.

---

## Summary

These decisions prioritize:
1. **Type Safety**: TOML encoding, explicit metadata, validation
2. **Extensibility**: Pluggable encoders, exception signaling, generic types
3. **Performance**: Lazy evaluation and smart fallback with bounded memory usage
4. **Usability**: Human-readable formats, clear error messages, backward compatibility
5. **Future-Proofing**: Versioning, modular structure, placeholder for remote backends
