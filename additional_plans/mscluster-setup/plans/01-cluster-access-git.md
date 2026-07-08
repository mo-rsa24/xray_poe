# 🔑 Cluster Access and Git Clone

## Description
Get Paper3 code onto mscluster110 by cloning the repo and checking out the working branch.

## Purpose
Without the repo on mscluster, nothing else can run. This is the first gate — O1 cluster-access+git.

## Goal
Repo cloned at `/home-mscluster/mmolefe/Playground/PhD/Paper3`, branch `auto/exec-plans-20260626`
checked out, and `git log --oneline -3` confirmed.

## Tasks
- [ ] ⚠️ Clone repo and checkout branch:
  ```
  cd /home-mscluster/mmolefe/Playground/PhD && git clone git@github.com:mo-rsa24/xray_poe.git Paper3 && cd Paper3 && git checkout auto/exec-plans-20260626
  ```
- [ ] ⚠️ Remedy (GitHub auth fails): reconnect with `ssh -A mscluster` (agent-forward) then retry clone;
  else generate a deploy key on mscluster (`ssh-keygen -t ed25519 -f ~/.ssh/id_deploy`) and add the public
  key to the repo's Deploy Keys; else switch to HTTPS:
  `git remote set-url origin https://github.com/mo-rsa24/xray_poe.git` then clone with a PAT.
- [ ] Verify: `git log --oneline -3` shows recent commits on `auto/exec-plans-20260626`.

## Engagement Instructions
Run on mscluster110:
```
git -C /home-mscluster/mmolefe/Playground/PhD/Paper3 log --oneline -3
git -C /home-mscluster/mmolefe/Playground/PhD/Paper3 branch --show-current
```
Expected: branch name `auto/exec-plans-20260626`, three recent commits visible. DoD-1 met.

## Recommended skill
— custom; no skill fits (project-specific cluster access and git setup).
