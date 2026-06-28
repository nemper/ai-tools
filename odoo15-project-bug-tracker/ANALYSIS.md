# Analysis & archival cleanup — `positive_bugs` (Project Bug Tracker)

Date: 2026-06-28. Reviewer: Claude (Opus 4.8).

## What the module does

An Odoo 15 addon that adds a bug/issue tracker to the Project app: a
`my.custom.bug.model` with a Kanban of custom stages (`my.custom.bug.stage`),
tags (`my.custom.bug.tag`), single/multi employee assignment, automatic
sequence-based Bug IDs, and timesheet integration via `account.analytic.line`.

## Problems found

1. **Bug — `write()` raised `AttributeError`.** The follower logic in `write()`
   used `record.assigned_to_id.partner_id` and
   `assigned_multi_user_ids.mapped('partner_id')`. `hr.employee` has **no**
   `partner_id` field in Odoo 15; the related partner is `user_id.partner_id`
   (the `create()` method already did this correctly). Any write touching an
   assigned bug would crash.

2. **Duplicated follower logic** in `create()` and `write()`, slightly different
   in each — the source of the bug above.

3. **License inconsistency.** `LICENSE` is MIT and the README says MIT, but the
   manifest declared `LGPL-3`.

4. **Manifest metadata placeholders.** `description` was `'...'`; category was
   `Uncategorized`.

5. **README rough edges.** Two stray `...` placeholder lines, a hardcoded
   `C:\Program Files\Odoo15\...` path, and no mention that the module's technical
   name must be `positive_bugs`.

6. **`create()` not batch-aware.** Used the legacy `@api.model` single-record
   signature instead of Odoo 15's recommended `@api.model_create_multi`.

## Changes made

- Extracted a single `_subscribe_assigned_employees()` helper used by both
  `create()` and `write()`, traversing `user_id.partner_id` — **fixes the
  `write()` crash** and removes the duplication.
- Modernized `create()` to `@api.model_create_multi` (batch `vals_list`),
  preserving the sequence-id and assignment behaviour.
- Manifest: `license` → `MIT` (matches `LICENSE`), real `description`, category
  `Project`, and a comment documenting the `positive_bugs` technical name.
- README: removed placeholder lines, dropped the hardcoded path, and added a
  **Deployment / Technical Name** section explaining the addon dir must be
  named `positive_bugs`.

## Not changed (deliberately)

- The fully-qualified `positive_bugs.*` XML IDs are kept (a Python `env.ref`
  requires a fully-qualified id). They are correct as long as the deploy folder
  is `positive_bugs`, now documented.
- Security CSV grants `base.group_user` full access — acceptable for this
  internal tool; left as-is.
