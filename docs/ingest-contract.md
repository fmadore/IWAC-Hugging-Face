# Omeka ingestion, privacy and recovery contract

The seven upload adapters write the **private complete mirror**. The public
dataset is produced by the public projection workflow. Review mapping changes
against `iwac_common/schema.py` (`SUBSETS`, `SOURCE_FIELD_COLUMNS`,
`DERIVED_FROM`) and the independently reviewed
`iwac_common/public_columns.json`. These in-repository files are the canonical
mapping contract; maintaining this project does not require an external skill.

## Startup settings

Settings load before repository defaults are constructed. Precedence is:

1. Explicit process environment (never overwritten by dotenv).
2. `IWAC_ENV_FILE`, when specified; otherwise `.env` discovered from the current
   working directory; otherwise `.env` in the workspace selected by
   `IWAC_WORK_DIR`, the source checkout, or the current directory for an
   installed wheel.
3. Documented production defaults.

`IWAC_HF_PRIVATE_REPO` and `IWAC_HF_PUBLIC_REPO` select repository destinations.
The upload parser reads the private destination lazily, so a programmatic
caller changing its environment before parsing gets the new destination.
`--repo` overrides that default. The write gateway independently enforces the
publication boundary; pointing a raw uploader at a public repository is not
an alternative publication mechanism.

Relative per-subset Omeka cache directories also resolve below that workspace.
An explicitly absolute cache path is preserved. This keeps runs launched from
different directories on the same configured workspace and keeps installed
package directories free of generated data.

A complete Omeka refresh requires both `OMEKA_KEY_IDENTITY` and
`OMEKA_KEY_CREDENTIAL`, including `--dry-run`. Missing either aborts before
fetching. An anonymous listing can omit private items and properties without
changing the visible count enough to trigger the shrink guard. The credentials
must belong to an account with the intended complete-source access; possession
of credentials by itself is not proof of that account's permissions.
`--help` and importing the package do not contact Omeka.

## Source visibility

Each adapter emits:

| Column | Meaning |
| --- | --- |
| `item_is_public` | True only when the parent has explicit `o:is_public: true`. Missing visibility is false. |
| `private_fields` | Sorted flat-column names containing values that are private or lack explicit visibility evidence. |
| `OCR_is_public` | True only when the parent is public and every nonempty content value is explicitly public. |

The complete mirror retains private values. The public projection removes
private parent rows and masks directly restricted source columns. A
multivalued or bilingual source property is treated conservatively: one
private value restricts all columns formed from that property. This includes
linked-resource IDs, dates and their numeric representations, coordinate axes,
descriptions, URLs, and live sentiment properties and justifications. A
private primary media resource also restricts its file/thumbnail/manifest
pointers. Retired annotation columns preserved from the baseline retain any
previously known field restrictions.

Existing explicitly approved computed derivatives follow the public policy in
`iwac_common/write_policy.py`; a source-private OCR does not automatically
withdraw every approved aggregate, embedding or topic output. OCR and
reconstructive lemma text remain restricted. A new column requires an explicit
rights review, not merely adding its name to the allowlist to silence an error.

Old datasets do not have sufficient parent/property visibility metadata. Run
authenticated ingestion for all subsets before the first publication with the
new policy. Do not fill unknown flags with `True` to bypass migration.

## Media and mapper failures

A failed mapper or media lookup aborts by default. Explicit
`--allow-map-failures` retains each failed item's entire baseline row and marks
it private until mapping succeeds; `--allow-media-failures` retains affected
media pointers in the private mirror and marks them restricted until a
successful refetch. A failure may reflect newly private upstream data, so old
visibility is not proof of current visibility. A primary PDF or image failure affects the file URL, thumbnail,
and IIIF manifest together. Neither option treats a transient network error as
evidence that previously accessible media disappeared. A failed new item with
no baseline cannot be reconstructed and still aborts.

## Derived values and durable recovery

Uploads invalidate stale derived cells **by default**. The source dependency
registry covers text, table of contents, language, image URL/thumbnail, and
publication date. Language changes invalidate language-dependent lemmas,
lexical/readability measures and topic assignments. Associated input/config
hashes are invalidated with their output. Mapper-produced values are already
fresh and are not overwritten by invalidation. Related-article rankings are
invalidated for the whole subset when candidate embeddings or corpus
membership change; a changed candidate can alter other articles' neighbours.

`--preserve-derived` explicitly retains stale cells. Before any remote write,
the uploader atomically stages affected IDs by derived column in
`.iwac_state/stale_derived/<owner>__<repo>__<subset>.json` (or
`IWAC_STATE_DIR`). Staging before the write preserves recovery information if
the process stops after a remote commit. A subsequent default upload or
`--invalidate-derived` replays the queue even when Omeka text already equals
the updated Hub text. Only a verified successful invalidating write clears
the queue. Failed writes retain it; dry runs neither stage nor delete it;
malformed or mismatched worklists fail closed.

A dedicated subset-state lock spans comparison, replay, staging, publication
and acknowledgment. It lives below the same state root, so two uploads sharing
a recovery queue cannot delete each other's newly staged work. This is
separate from the short repository commit lock and the remote expected-revision
check. A dry run may create the temporary lock but cannot alter its worklist.

After an invalidating upload, rerun the affected enrichment stages. Text/image
embedding and lemma provenance distinguishes input changes from configuration
changes and permits selective recomputation. A stale-worklist replay is
conservative: if an enrichment was recomputed independently while its queue
was retained, replay may clear it again. Resolve pending ingest queues before
running expensive enrichment, and retain the state directory across hosts or
use the default automatic invalidation so recovery does not depend on copying
local state.

## Deleted references

References now default to `--stale-rows drop`, matching source absence after a
complete authenticated fetch. The existing shrink guard still requires
`--force-shrink` for a substantial intentional deletion. For historical work,
`--stale-rows keep` retains the complete baseline record in the private mirror
and marks it `item_is_public=False`; it does not fabricate a row with empty
bibliographic fields or republish a deleted source record.

The current default cannot reconstruct source changes that were already
uploaded by an older version without a retained stale worklist or provenance.
For such historical runs, recompute the affected enrichments from a known
source revision and retain the new provenance.
