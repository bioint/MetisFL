## How to create a pull request on bitbucket.org.
1. create a new branch using any of the four branch types: `release/, hotfix/, feature/, bugfix/`     
   e.g., `git checkout -b feature/\<branch_suffix\>`
2. make all the necessary changes/edits you want on the project and commit
3. push the local branch to the remote
4. navigate on the bitbucket environment, find your branch (left sidebar -\> branches -\> your_new_branch)
5. create a pull request and assign reviewers
6. if changes need to be made on the request, then work on your local branch, commit and push
7. reviewers will see the new edits and if approved then they will merge the request

##### Recommendation on how to identify your branch type:
You need to figure out the type of your patch. For instance, if a bunch of modules are bundled together
then this is a release; if your patch performs minor edits/alterations to an existing approach but the
functionality and the logic of the codebase remains the same, then you can assign it as a hotfix; If you
introduce an entirely new implementation, functionality then feature is the best option; if you fix severe
bugs then of course bugfix is where you need to push.  

## How to merge pull requests.
Let's see an example where we try to merge the edits we made on the feature branch (source) into the master branch (destination).
The destination branch is where we want to apply our changes and feature branch is our current working files.

##### Perform Merge and Resolve Conflicts:
https://support.atlassian.com/bitbucket-cloud/docs/resolve-merge-conflicts/

Steps:
1. git checkout \<destination\>
2. git pull
3. git checkout \<source\>
4. git merge \<destination\>
This is where the magic happens; there will be conflicts and we need to make the appropriate edits.
When you merge two branches with conflicts locally, you'll get conflict markers in the file when you open your editor.
Open the file to resolve the conflict. You can do this using the command line or you can navigate to the file.
5. git add \<modified_filenames\>
6. git commit -m 'fixed conflicts with \<destination\> branch'
7. git push origin \<source\>

Example:

````
git checkout dev
git pull
git checkout feature/ModifyInt32TypeToUint32
git merge dev
<resolve confilcts>
git add <modified_files>
git commit -m ""
git push origin feature/ModifyInt32TypeToUint32
<press merge in the Bitbucket UI>
````

##### Merge Strategies:

https://support.atlassian.com/bitbucket-cloud/docs/merge-a-pull-request/#Mergeapullrequest-Mergestrategies
  
- Merge commit (--no-ff): Always create a new merge commit and update the \<destination\> branch to it, even if the \<source\> branch is already up to date with the \<destination\> branch.

- Fast-forward (--ff): If the \<source\> branch is out of date with the \<destination\> branch, create a merge commit. Otherwise, update the \<destination\> branch to the latest commit on the \<source\> branch.

- Fast-forward only (--ff-only): If the \<source\> branch is out of date with the \<destination\> branch, reject the merge request. Otherwise, update the \<destination\> branch to the latest commit on the \<source\> branch.

- Rebase, merge  (rebase + merge --no-ff): Commits from the \<source\> branch onto the \<destination\> branch, creating a new non-merge commit for each incoming commit. Creates a merge commit to update the \<destination\> branch. The PR branch is not modified by this operation.

- Rebase, fast-forward (rebase + merge --ff-only): Commits from the \<source\> branch onto the \<destination\> branch, creating a new non-merge commit for each incoming commit. Fast-forwards the \<destination\> branch with the resulting commits. The PR branch is not modified by this operation.

- Squash (--squash) **METIS DEFAULT**: Combine all commits into one new non-merge commit on the \<destination\> branch.

- Squash, fast-forward only (--squash --ff-only): If the \<source\> branch is out of date with the \<destination\> branch, reject the merge request. Otherwise, combine all commits into one new non-merge commit on the \<destination\> branch.


## Bazel files formatting conventions:
- For lists with a single element, we do not go next line.
- For lists with at least 2 elements, we start the first element from the line below.