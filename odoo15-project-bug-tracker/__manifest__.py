{
    'name': 'My Custom Module',
    'version': '1.0',
    'summary': 'A custom module with a blank form',
    'description': 'This module creates a blank form accessible from the menu.',
    'author': 'Your Name',
    'category': 'Uncategorized',
    'depends': ['base', 'contacts', 'project'],
    'data': [
        'security/ir.model.access.csv',
        'views/my_custom_view.xml',
        'data/bug_stage_data.xml',
    ],
    'installable': True,
    'application': True,
    'license': 'LGPL-3',  # Adding the license key to avoid warnings
}