{
    'name': 'My Custom Module',
    'version': '2.0',
    'summary': 'Bug tracking module for Projects',
    'description': '...',
    'author': 'Nemanja Perunicic',
    'category': 'Uncategorized',
    'depends': ['base', 'contacts', 'project'],
    'data': [
        'security/ir.model.access.csv',
        'views/my_custom_view.xml',
        'data/bug_stage_data.xml',
    ],
    'installable': True,
    'application': True,
    'license': 'LGPL-3',
}