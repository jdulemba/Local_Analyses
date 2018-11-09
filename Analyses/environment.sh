export ANALYSES_PROJECT=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P)
#source ANALYSES env
git_dir=$(basename $(dirname $ANALYSES_PROJECT))

branch=$( git symbolic-ref --short HEAD )
echo "On branch '$branch' of $git_dir"


export jobid=2018Nov08
echo "jobid=$jobid"
