## GitHub Workflow 

### Clone the repo
```bash
git clone https://github.com/maxim-pre/RL1_CW.git
cd RL1_CW
```

### 1. Work from dev branch
```bash
git checkout dev # move to the dev branch
git pull origin dev # get most up to date code from repo
```

### 2. Create your own branch from the dev branch
```bash
git checkout -b branch_name # create new branch 
```
### 3. Make changes and commit
```bash
git add .
git commit -m "comment"
```
### 4. Stay up to date 
```bash
git fetch origin
git rebase origin/dev 
```
### 5. push your branch to github
```bash
git push origin branch_name
```

### 6. open pull request
```
base branch: dev
compare branch: branch_name
```

### 7. cleanup after merge
```bash
git checkout dev
git pull origin dev
git branch -d branch_name
git push origin --delete branch_name

create new branch again
```


