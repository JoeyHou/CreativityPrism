# Stage all changes
git add -A

# Commit them
git commit -m "Update leaderboard styling and tabs"

# Push to your repo (main branch)
git push origin main   # or master if that's your default branch

# Deploy site
mkdocs gh-deploy
