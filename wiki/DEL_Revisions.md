# Delete GitHub wiki revisions

```python
# Delete prior revisions from a GitHub wiki so that only the most-recent
# version of the content is available.

# Clone the wiki.
git clone https://github.com/[user]/[repo].wiki.git

# Remove the .git folder.
rm -rf .git

# Reconstruct the local repo with only latest content
git init
git add .
git commit -m "Initial commit"

# Push to GitHub
git remote add origin <github-uri>
git push -u --force origin master
```

## Refer

[https://gist.github.com/hacksalot/72517b9b1c145116e89e](https://gist.github.com/hacksalot/72517b9b1c145116e89e)
