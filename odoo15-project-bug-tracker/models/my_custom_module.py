import logging
from odoo import models, fields, api

# Initialize logger for this module
_logger = logging.getLogger(__name__)

# Define the MyCustomModel class inheriting from Odoo's models.Model
class MyCustomModel(models.Model):
    _name = 'my.custom.model'  # Define the model name
    _description = 'My Custom Model'  # Model description

    # Define fields for the model
    company_id = fields.Many2one('res.partner', string="Company", domain="[('is_company', '=', True)]")
    # Many2one field to select a company from res.partner, with a domain filtering only companies

    company_address = fields.Char(string="Address", readonly=True)
    # Field to display the company's address, read-only

    company_phone = fields.Char(string="Phone", readonly=True)
    # Field to display the company's phone number, read-only

    project_list = fields.Text(string="Projects", readonly=True)
    # Field to display the list of projects related to the company and its employees, read-only

    project_number = fields.Integer(string="Number of Projects", readonly=True)
    # Field to display the number of projects related to the company and its employees, read-only

    @api.onchange('company_id')
    def _onchange_company_id(self):
        # Method called when the company_id field value changes
        if self.company_id:
            # If a company is selected, set values for the company's address and phone number
            self.company_address = self.company_id.contact_address
            self.company_phone = self.company_id.phone
            _logger.info('Selected Company: %s (ID: %s)', self.company_id.name, self.company_id.id)

            # Search for projects related to the selected company and its employees
            projects = self.env['project.project'].search([('partner_id', '=', self.company_id.id)])
            employees = self.env['res.partner'].search([('parent_id', '=', self.company_id.id)])
            for employee in employees:
                employee_projects = self.env['project.project'].search([('partner_id', '=', employee.id)])
                projects |= employee_projects

            _logger.info('Projects domain search for company and employees: %s', projects)

            if not projects:
                # If no projects are found, set project_number to 0
                _logger.info('No projects found for company ID: %s', self.company_id.id)
                self.project_number = 0
            else:
                # If projects are found, set project_number to the number of found projects
                _logger.info('Projects found: %s', projects)
                self.project_number = len(projects)

            # Create a dictionary to store projects by owner (company or employee)
            projects_by_owner = {}
            projects_by_owner[self.company_id.name] = projects.filtered(lambda p: p.partner_id == self.company_id)
            for employee in employees:
                projects_by_owner[employee.name] = projects.filtered(lambda p: p.partner_id == employee)

            # Log information about found projects and construct the project_list field value
            project_list = []
            for owner, proj in projects_by_owner.items():
                if proj:
                    project_list.append(f"{owner}:")
                    project_list.extend(proj.mapped('name'))
                    project_list.append('')  # Add an empty line
                    _logger.info('Owner: %s, Projects: %s', owner, proj.mapped('name'))

            self.project_list = '\n'.join(project_list).strip()  # Remove the trailing empty line
        else:
            # If no company is selected, set fields to empty values
            self.company_address = ''
            self.company_phone = ''
            self.project_list = ''
            self.project_number = 0
            