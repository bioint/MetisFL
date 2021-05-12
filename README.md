## Prerequisites
- Install googletest (MacOS as `brew install googletest`)
- Install protobuf (MacOS as `brew install protobuf`)


## Trello UI
https://trello.com/b/bYLUYqGK/metis-v01

## How to create a pull request on bitbucket.org!
1. create a new branch using any of the four branch types: `release/, hotfix/, feature/, bugfix/`     
   e.g., `git checkout -b feature/<branch_suffix>`
2. make all the necessary changes/edits you want on the project and commit
3. push the local branch to the remote
4. navigate on the bitbucket environment, find your branch (left sidebar -> branches -> your_new_branch)
5. create a pull request and assign reviewers
6. if changes need to be made on the request, then work on your local branch, commit and push
7. reviewers will see the new edits and if approved then they will merge the request

##### Recommendation on how to identify your branch type:
You need to figure out the type of your patch. For instance, if a bunch of modules are bundled together 
then this is a release; if your patch performs minor edits/alterations to an existing approach but the 
functionality and the logic of the codebase remains the same, then you can assign it as a hotfix; If you
introduce an entirely new implementation, functionality then feature is the best option; if you fix severe
bugs then of course bugfix is where you need to push.  

