{
    'name': 'Project Bug Tracker',
    'version': '15.0.0.0.1',
    'summary': 'Bug tracking module for Projects',
    'description': '...',
    'author': 'Nemanja Perunicic',
    'category': 'Uncategorized',
    'depends': ['base', 'contacts', 'project'],
    'data': [
        'security/ir.model.access.csv',
        'views/my_custom_view.xml',
        'data/bug_stage_data.xml',
        'data/bug_sequence.xml',
    ],
    'installable': True,
    'application': True,
    'license': 'LGPL-3',
}