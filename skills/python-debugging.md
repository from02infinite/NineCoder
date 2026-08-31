# Python Debugging Skill

Use this skill when fixing Python bugs.

Process:
1. Reproduce the failure with a small command or test.
2. Read the smallest relevant source file.
3. Make one targeted edit.
4. Rerun the same command.
5. Add an edge-case check when cheap.

Prefer exact `edit_file` replacements over whole-file rewrites.
