# NOTE: The module's *technical name* is `positive_bugs` (several XML IDs and an
# `env.ref('positive_bugs....')` call depend on it). When deploying, the addon
# directory MUST be named `positive_bugs`, regardless of this repository folder
# name. See README.md ("Deployment / technical name").
{
    'name': 'Project Bug Tracker',
    'version': '15.0.0.2.6',
    'summary': 'Bug tracking module for Projects',
    'description': """
Project Bug Tracker
===================
Adds a dedicated bug/issue tracking model to the Odoo Project module.

Features:
- Bugs linked to projects and tasks, with a Kanban workflow of custom stages.
- Single and multi-employee assignment; assignees are auto-subscribed as followers.
- Automatic sequence-based human-readable Bug ID (``BUG/00001``).
- Timesheet integration (``account.analytic.line``) for effort tracking per bug.
- Priority/urgency levels, tags, version tracking and rich description fields.
""",
    'author': 'Nemanja Perunicic',
    'category': 'Project',
    'depends': ['base', 'contacts', 'project', 'hr_timesheet', ],
    'data': [
        'security/ir.model.access.csv',
        'views/my_custom_view.xml',
        'data/bug_stage_data.xml',
        'data/bug_sequence.xml',
        'actions/actions.xml',
        'views/project_task_views_inherit.xml',
        'views/my_custom_bug_stage_views.xml',
    ],
    'installable': True,
    'application': True,
    # Matches the bundled LICENSE file (MIT). Odoo 15's license selection does not
    # list MIT, so the Apps UI may show it as "Other"; this is informational only.
    'license': 'MIT',
}