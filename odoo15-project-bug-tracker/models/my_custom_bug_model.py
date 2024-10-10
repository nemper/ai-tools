import random
from odoo import models, fields, api, _
import logging
_logger = logging.getLogger(__name__)


class MyCustomBugModel(models.Model):
    _name = 'my.custom.bug.model'
    _description = 'My Custom Bug Model'
    _order = 'bug_unique_id desc'
    _inherit = ['mail.thread', 'mail.activity.mixin']

    # Ensure stage_id is properly defined
    stage_id = fields.Many2one('my.custom.bug.stage', string="Stage", group_expand='_read_group_stage_ids', default=lambda self: self.env['my.custom.bug.stage'].search([], limit=1))
    name = fields.Char(string="Bug Name", required=True)
    project_id = fields.Many2one('project.project', string="Project")
    task_id = fields.Many2one('project.task', string="Task", domain="[('project_id', '=', project_id)]")
    assigned_to_id = fields.Many2one('res.users', string="Assigned to")
    assigned_multi_user_ids = fields.Many2many('res.users', string="Assign Multi User")
    client_id = fields.Many2one('res.partner', string="Client")
    priority = fields.Selection([
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High')
    ], string="Bug Priority", default='medium')
    tag_ids = fields.Many2many('my.custom.bug.tag', string="Tags")
    submitted_by_id = fields.Many2one('res.users', string="Submitted by")
    version = fields.Char(string="Version")
    deadline = fields.Date(string="Deadline")
    issue_type = fields.Selection([
        ('data', 'Data'),
        ('ui', 'UI'),
        ('backend', 'Backend'),
        ('frontend', 'Frontend')
    ], string="Issue Type")
    build_date = fields.Date(string="Build Date")
    issue_resides_in = fields.Selection([
        ('backend', 'Backend'),
        ('frontend', 'Frontend'),
    ], string="Issue Resides in")
    client_urgency = fields.Selection([
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High')
    ], string="Client Urgency", default='medium')
    fixed_version = fields.Char(string="Fixed Version")
    active = fields.Boolean(string="Active", default=True)

    bug_unique_id = fields.Char(string="Bug ID", readonly=True, copy=False, index=True, default='Auto-generated after Save')

    description = fields.Text(string="Description")
    steps_to_replicate = fields.Text(string="Steps to Replicate")
    expected_result = fields.Text(string="Expected Result")
    actual_result = fields.Text(string="Actual Result")
    solution = fields.Text(string="Solution")
    extra_info = fields.Text(string="Extra Info")

    @api.onchange('assigned_to_id')
    def _onchange_assigned_to_id(self):
        if self.assigned_to_id and self.assigned_to_id not in self.assigned_multi_user_ids:
            self.assigned_multi_user_ids |= self.assigned_to_id

    @api.model
    def create(self, vals):
        _logger.info('Creating a new bug with values: %s', vals)
        if not vals.get('bug_unique_id') or vals['bug_unique_id'] == 'Auto-generated after Save':
            max_id = self.search(
                [('bug_unique_id', '!=', 'Auto-generated after Save')],
                order='bug_unique_id desc',
                limit=1
            ).bug_unique_id
            if max_id and max_id.isdigit():
                vals['bug_unique_id'] = str(int(max_id) + 1)
            else:
                vals['bug_unique_id'] = '1'
        record = super(MyCustomBugModel, self).create(vals)
        if record.assigned_to_id and record.assigned_to_id not in record.assigned_multi_user_ids:
            record.assigned_multi_user_ids |= record.assigned_to_id
        # Add assigned users as followers
        partners_to_subscribe = []
        if record.assigned_to_id:
            partners_to_subscribe.append(record.assigned_to_id.partner_id.id)
        if record.assigned_multi_user_ids:
            partners_to_subscribe.extend(record.assigned_multi_user_ids.mapped('partner_id').ids)
        if partners_to_subscribe:
            record.message_subscribe(partners_to_subscribe)
        return record

    @api.model
    def _read_group_stage_ids(self, stages, domain, order):
        # Define how the stages are grouped and ordered in the Kanban view
        stage_ids = stages.search([], order=order)
        return stage_ids
    
    def write(self, vals):
        res = super(MyCustomBugModel, self).write(vals)
        for record in self:
            # Ensure assigned_to_id is in assigned_multi_user_ids
            if 'assigned_to_id' in vals and record.assigned_to_id and record.assigned_to_id not in record.assigned_multi_user_ids:
                record.assigned_multi_user_ids |= record.assigned_to_id
            # Add assigned users as followers
            partners_to_subscribe = []
            if record.assigned_to_id:
                partners_to_subscribe.append(record.assigned_to_id.partner_id.id)
            if record.assigned_multi_user_ids:
                partners_to_subscribe.extend(record.assigned_multi_user_ids.mapped('partner_id').ids)
            if partners_to_subscribe:
                record.message_subscribe(partners_to_subscribe)
        return res


class MyCustomBugTag(models.Model):
    _name = 'my.custom.bug.tag'
    _description = 'Bug Tag'

    name = fields.Char(string="Tag Name", required=True)
    color = fields.Integer(string="Color")  # Add this field to store the color