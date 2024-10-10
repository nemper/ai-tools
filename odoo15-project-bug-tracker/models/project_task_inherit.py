# project_task_inherit.py
from odoo import models, fields, api

class ProjectTask(models.Model):
    _inherit = 'project.task'

    bug_count = fields.Integer(
        string='Bug Count',
        compute='_compute_bug_count'
    )

    def _compute_bug_count(self):
        for task in self:
            task.bug_count = self.env['my.custom.bug.model'].search_count([('task_id', '=', task.id)])

    def action_view_bugs(self):
        action = self.env.ref('positive_bugs.action_my_custom_bug_model').read()[0]
        action['domain'] = [('task_id', '=', self.id)]
        action['context'] = {
            'default_task_id': self.id,
            'search_default_task_id': self.id,
        }
        return action
