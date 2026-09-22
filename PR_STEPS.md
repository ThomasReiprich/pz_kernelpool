# From the built submission to the pull request (step by step)

Everything below is done once, in this order. Commands run in a terminal on
the Mac, in the DESC_NZ_Challenge folder, with `conda activate desc-pz`.
Nothing here touches the organizers' repository until step 6. The GitHub
user name is `ThomasReiprich`; the placeholders in the harness files are
already filled with it, so the release asset's URL is fixed in advance:

    https://github.com/ThomasReiprich/pz_kernelpool/releases/download/v1.0.0/kernelpool_submission.tgz

## 1. Install the package and build the submission (~1 h 15)

    pip install -e submission
    bash submission/build_submission.sh

Ends with "all checks passed", `submissions/kernelpool_submission.tgz`
(about 1.1 GB: eight submission files + eight model bundles +
`build_info.json`) and `submission/environment_desc-pz.txt` (the `pip
freeze` of the environment the models were built in). The pre-made files
are in `submissions/kernelpool/`, the work directories in `pilot/final/`.
The script stops before the tarball if any check fails.

## 2. Pin the model-relevant packages

After the build, Claude sets the versions of `pz-rail-flexzboost`,
`pz-rail-gpz_v1`, `flexcode`, `xgboost` and `qp-prob` in
`submission/pyproject.toml` from `environment_desc-pz.txt` (these decide
whether the pickled models load); the other dependencies stay unpinned so
that the organizers' runner can resolve them next to the harness. Then
Claude tests the install from GitHub in a fresh environment (step 4).

## 3. Put the package on GitHub (public repository `pz_kernelpool`)

The organizers' CI installs our method with `pip install
git+https://github.com/ThomasReiprich/pz_kernelpool.git@v1.0.0` and
downloads the tarball from a *release* of that repository, so both need to
be public.

On github.com: "New repository", name `pz_kernelpool`, public, no
README (we have one). Then, in the terminal:

    cd submission
    git init -b main
    git add pyproject.toml README.md PR_STEPS.md build_submission.sh environment_desc-pz.txt pz_method_writeup.md kernelpool pzdc
    git commit -m "kernelpool 1.0.0: PZ data challenge entry for task sets 3 and 4"
    git remote add origin https://github.com/ThomasReiprich/pz_kernelpool.git
    git push -u origin main
    git tag v1.0.0
    git push origin v1.0.0
    cd ..

(`git push` asks for your GitHub user name and a personal access token
in place of the password; GitHub → Settings → Developer settings →
Personal access tokens → "Tokens (classic)", scope `repo`. `git add` with
a directory name adds everything inside it; `__pycache__` folders are
excluded by the `.gitignore` in `submission/`.)

Then upload the tarball as a release asset: on github.com, in
`pz_kernelpool`: "Releases" → "Draft a new release" → choose tag
`v1.0.0` → title "kernelpool 1.0.0" → drag
`submissions/kernelpool_submission.tgz` into the assets box → "Publish
release". Check that the URL at the top of this file downloads the file
(open it in a browser).

## 4. Fresh-environment test (Claude, in the cloud sandbox)

`pip install -r pzdc/requirements_kernelpool.txt` into a fresh Python
3.13 environment, then `estimate_only` on one bundle from the release
(the ts3 Cardinal 1yr one) against its test file; the result must match
the submitted file. This is the check that the pickled models load
outside the environment they were built in.

## 5. Test the harness locally (optional but recommended, ~2 h)

    cd repos/pz_data_challenge
    git checkout -b submit/kernelpool
    cp ../../submission/pzdc/test_kernelpool.py tests/
    cp ../../submission/pzdc/requirements_kernelpool.txt .
    cp ../../submission/pzdc/submit_kernelpool.yaml .github/workflows/
    pip install -r requirements_kernelpool.txt
    ln -s ../../../data/pz/public tests/public       # the public data are already on disk
    NO_TEARDOWN=1 PZDC_CI_MAX_TRAIN=6000 python -m pytest tests/test_kernelpool.py -x -s

This downloads the tarball, validates the eight files, runs the
estimation-only path from the bundles and a subsampled retrain for every
combination. `SHORT_TASKS_23=1` restricts the two run paths to Cardinal
10yr if time is short.

## 6. Open the pull request

The organizers' repository is `LSSTDESC/pz_data_challenge`; you need a
fork of it under your account (github.com → the repository page →
"Fork"). Then:

    cd repos/pz_data_challenge
    git remote add fork https://github.com/ThomasReiprich/pz_data_challenge.git
    git add tests/test_kernelpool.py requirements_kernelpool.txt .github/workflows/submit_kernelpool.yaml
    git commit -m "Submit/kernelpool: task sets 3 and 4 (Bonn)"
    git push -u fork submit/kernelpool

On github.com, the fork's page shows "Compare & pull request" for the
new branch; the PR goes to `LSSTDESC/pz_data_challenge`, base `main`.
Title "Submit/kernelpool (task sets 3 and 4)"; body: the method
description (`pz_method_writeup.md`), which is one page. The CI runs the
same pytest as step 5 with `PZDC_CI_MAX_TRAIN=6000`.

## What can go wrong

The tarball must extract to files at the top level of
`submissions/kernelpool/` (the build script makes it so). The release
must be public. If the CI runner exceeds its time, add
`SHORT_TASKS_23: "1"` to the workflow's `env:` block. If `pip install`
of the package fails on the runner, the pinned versions in
`pyproject.toml` are the first thing to check against the runner's
Python (3.13 in the template); step 4 is meant to catch that first.
