from pathlib import Path
from setuptools import find_packages, setup

HERE = Path(__file__).parent


def read_requirements(path=HERE / "requirements.txt"):
	text = path.read_text(encoding="utf-8")
	reqs = []
	for line in text.splitlines():
		line = line.strip()
		if not line or line.startswith("#"):
			continue
		reqs.append(line)
	return reqs


long_description = ""
if (HERE / "README.md").exists():
	long_description = (HERE / "README.md").read_text(encoding="utf-8")


setup(
	name="arise_chatbot",
	version="0.1.0",
	description="Arise.ai chatbot utilities and demo code",
	long_description=long_description,
	long_description_content_type="text/markdown",
	author="Bhaveshsisodia",
	license="MIT",
	url="https://github.com/Bhaveshsisodia/Arise.ai",
	packages=find_packages(exclude=("tests",)),
	include_package_data=True,
	python_requires=">=3.10",
	install_requires=read_requirements(),
	classifiers=[
		"Programming Language :: Python :: 3",
		"Programming Language :: Python :: 3.10",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
)

