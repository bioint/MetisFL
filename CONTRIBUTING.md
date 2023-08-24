
To contribute to this GitHub project, you can follow these steps:

1. Fork the repository you want to contribute to by clicking the "Fork" button on the project page.

2. Clone the repository to your local machine and enter the newly created repo using the following commands:

```
git clone https://github.com/YOUR-GITHUB-USERNAME/metisfl.git
cd metisfl
```
3. Add metisfl original repository as upstream, to easily sync with the latest changes.

```
git remote add upstream https://github.com/NevronAI/metisfl.git
```

4. Create a new branch for your changes using the following command:

```
git checkout -b "branch-name"
```
5. Make your changes to the code or documentation.

6. Add the changes to the staging area using the following command:
```
git add . 
```

7. Commit the changes with a meaningful commit message using the following command:
```
git commit -m "your commit message"
```
8. Push the changes to your forked repository using the following command:
```
git push origin branch-name
```
9. Go to the GitHub website and navigate to your forked repository.

10. Click the "New pull request" button.

11. Select the branch you just pushed to and the branch you want to merge into on the original repository.

12. Add a description of your changes and click the "Create pull request" button.

13. Wait for the project maintainer to review your changes and provide feedback.

14. Make any necessary changes based on feedback and repeat steps 5-12 until your changes are accepted and merged into the main project.

15. Once your changes are merged, you can update your forked repository and local copy of the repository with the following commands:

```
git fetch upstream
git checkout main
git merge upstream/main
```
Finally, delete the branch you created with the following command:
```
git branch -d branch-name
```
That's it you made it 🐣⭐⭐
