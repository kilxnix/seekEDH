# Repository Access Rules

This project includes hooks that disable remote access. Use them if you need to prevent others from pushing or pulling code.

## Block pushes

Copy `hooks/pre-receive` into your repository's `.git/hooks/` directory on the server and make sure it is executable. Any attempt to push will be rejected with an error message.

```
cp hooks/pre-receive .git/hooks/pre-receive
chmod +x .git/hooks/pre-receive
```

## Block pulls

To prevent fetching or cloning, configure Git to use the provided `hooks/deny-upload-pack.sh` script as the upload-pack command:

```
cp hooks/deny-upload-pack.sh .git/hooks/
chmod +x .git/hooks/deny-upload-pack.sh
git config --local core.uploadpack "$(pwd)/.git/hooks/deny-upload-pack.sh"
```

After this configuration, all pull or clone attempts will fail with a message from the script.
