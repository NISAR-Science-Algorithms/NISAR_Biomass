# NISAR_Biomass


Repository for L3 science products for the Ecosystems Biomass workflow




### Installation and Setup:
1) Fork the repository

2) Clone your fork to your local machine with an SSH key
   ```

   git clone git@github.com:{your_github_username}/NISAR_Biomass.git

   ```
3) Install the required Python packages
   ```
   cd NISAR_Biomass
   conda env create -f requirements.yml
   conda activate NISAR_Biomass
   ```
4) Run the notebooks
   ```
   jupyter notebook
   ```
test

   
### For Developers Submitting Code
1) Install pre-commit to ensure pre-commit hooks are run
   ```
   pip install pre-commit
   pre-commit install
   ```
2) Install Trufflehog

   To install in a specific directory, change ~/ to your chosen path
   ```
   curl -sSfL https://raw.githubusercontent.com/trufflesecurity/trufflehog/main/scripts/install.sh | sh -s -- -b ~/
   ```
   Make sure to add the path to trufflehog to your $PATH.
   ```
   export PATH="$PATH:~"
   ```
   
4) Make a new branch to develop your changes in
   ```
   git checkout -b {your_branch}
   ```
5) Make changes to files and add changes to your commit
   ```
   git add {file}
   ```
   ***Make sure to clear the outputs of any Jupyter Notebook before committing changes.***
7) Commit changes to your fork
   ```
   git commit -m "comments related to this commit"
   ```
   This will run trufflehog pre-commit hooks to scan for any potential secrets. If secrets are detected, this will fail and you will need to resolve the issues
8) Push your commit to your branch in your fork
   ```
   git push --set-upstream origin {your_branch}
   ```
9) Go back to your fork on Github.com and submit a merge request
    
