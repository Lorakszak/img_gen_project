""" Fixing requirements.txt making sure all libraries contain their versions instead of reference to the build from source aka '@ file:///home/cuda...'"""

import re
import subprocess


def get_installed_version(package_name: str) -> None:
    """Get the installed version of the package using pip."""
    try:
        # Run pip show to get package details
        result = subprocess.run(
            ["pip", "show", package_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        # Parse the version from the output
        for line in result.stdout.splitlines():
            if line.startswith("Version:"):
                return line.split(" ")[1]
        return None
    except subprocess.CalledProcessError:
        print(f"Failed to get version for package: {package_name}")
        return None


def fix_requirements_file(file_path: str) -> None:
    with open(file_path, "r") as f:
        lines = f.readlines()

    updated_lines = []

    for line in lines:
        # Check if the line contains @ file:// pattern
        if "@ file://" in line:
            # Extract the package name before @
            package_name = re.split(r"\s+@", line)[0]
            # Get the installed version
            installed_version = get_installed_version(package_name)

            if installed_version:
                # Replace the line with package==version
                updated_line = f"{package_name}=={installed_version}\n"
                print(f"Updating: {line.strip()} -> {updated_line.strip()}")
                updated_lines.append(updated_line)
            else:
                # Keep the original line if version is not found
                updated_lines.append(line)
        else:
            # Keep the original line for packages that already have proper version
            updated_lines.append(line)

    # Write the updated requirements back to the file
    with open(file_path, "w") as f:
        f.writelines(updated_lines)


if __name__ == "__main__":
    # Specify the path to your requirements.txt file
    requirements_file = "requirements.txt"
    fix_requirements_file(requirements_file)
