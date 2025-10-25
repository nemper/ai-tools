from odoo import models, fields

class MyCustomBugStage(models.Model):
    _name = 'my.custom.bug.stage'
    _description = 'Bug Stage'
    _order = 'sequence, id'

    name = fields.Char(string="Stage Name", required=True, translate=True)
    sequence = fields.Integer(string="Sequence", default=10)
    fold = fields.Boolean(string="Folded in Kanban", default=False)
