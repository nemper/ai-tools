# Custom Bug Tracker for Odoo 15 Project Module

## Overview

This Odoo 15 module introduces a custom bug tracking system integrated within the Project module. It allows organizations to efficiently manage bugs, feature requests, and improvements directly linked to projects and tasks. The module enhances collaboration by enabling bug assignment to employees, tracking progress through customizable stages, and integrating timesheets for effort tracking.

## Features

- **Bug Management within Projects**: Create and manage bug reports associated with specific projects and tasks.

- **Assignment and Collaboration**:
  - Assign bugs to single or multiple employees.
  - Automatically add assigned employees as followers for seamless communication.

- **Customizable Stages**:
  - Define and customize bug stages to reflect your workflow.
  - Kanban view with grouping by stages for visual management.

- **Timesheet Integration**:
  - Link timesheet entries to bugs for accurate time tracking.
  - View and manage timesheets directly from the bug form.

- **Automatic Bug Numbering**:
  - Each bug is assigned a unique identifier (`bug_unique_id`) using an Odoo sequence.

- **Detailed Bug Reporting**:
  - Capture comprehensive bug details, including:
    - Description
    - Steps to replicate
    - Expected result
    - Actual result
    - Solution
    - Additional information

- **Priority and Urgency Levels**:
  - Set bug priority (`Low`, `Medium`, `High`).
  - Indicate client urgency to prioritize resolution.

- **Version Tracking**:
  - Record the affected version and fixed version of the project.
  - Log build dates for reference.

- **Tagging System**:
  - Categorize bugs using custom tags with color coding.

- **Deadline Management**:
  - Set deadlines for bug resolution to ensure timely fixes.

...

## Installation

1. **Prerequisites**:
   - Odoo 15 installed with the Project and Timesheet modules.
   - Access to the Odoo server file system to add custom modules.

2. **Download the Module**:
   - Clone or download this repository to your local machine.
   - C:\Program Files\Odoo15\server\odoo\addons

3. **Add to Odoo Addons Path**:
   - Copy the module folder to your Odoo addons directory.

4. **Update Module List**:
   - Restart the Odoo server.
   - Navigate to `Apps` in Odoo.
   - Click on `Update Apps List`.

5. **Install the Module**:
   - Search for `Bug Tracker` in the Apps.
   - Click `Install`.

## Usage

1. **Accessing the Bug Tracker**:
   - Navigate to the `Project` module.
   - Under the `Bugs` menu, select `Manage Bugs`.

2. **Creating a New Bug**:
   - Click on `Create`.
   - Fill in the necessary fields:
     - **Bug Name**: Title of the bug.
     - **Bug Type**: Select from Bug, Improvement, User Request, or New Functionality.
     - **Project**: Link the bug to a project.
     - **Task**: Optionally link to a specific task.
     - **Assigned To**: Assign the bug to an employee.
     - **Priority** and **Client Urgency**: Set the importance levels.
     - **Deadline**: Set a resolution deadline.
     - **Description** and other detailed fields.

3. **Managing Bugs**:
   - Use the Kanban view to drag and drop bugs between stages.
   - Filter and group bugs based on various criteria.
   - Communicate with team members using the chatter.

4. **Timesheet Entries**:
   - Log time spent on bugs through the `Timesheets` tab in the bug form.
   - Timesheet entries are linked to both the bug and the associated project/task.

5. **Tags and Stages**:
   - Manage tags via the `Edit Tags` menu under `Bugs`.
   - Customize bug stages as needed for your workflow.

...

## Configuration

- **Sequences**:
  - The module uses an Odoo sequence for generating unique bug IDs.
  - Ensure the sequence `my.custom.bug.model` is configured if you wish to customize numbering.

- **User Access Rights**:
  - By default, only employees can be assigned to bugs.
  - Adjust access rights and record rules in the `security` directory if necessary.

## Dependencies

- **Odoo 15**: Core Odoo installation.
- **Project Module**: Standard Odoo module for project management.
- **Timesheet Module**: Standard Odoo module for time tracking.

## Folder Structure

- **actions**: Server actions and automation scripts.
- **data**: Data files like sequences and initial configurations.
- **models**: Python models defining new data structures.
- **security**: Access control and security rules.
- **static/description**: Module images and descriptions.
- **views**: XML files defining form, tree, and kanban views.
- **\_\_init\_\_.py**: Python file initializing the module.
- **\_\_manifest\_\_.py**: Module manifest file with metadata.
- **LICENSE**: License information for the module.

## License

This module is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
