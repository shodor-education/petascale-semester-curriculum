#!/bin/bash
#=================================================================================================
# This is a basic Bash script that can be used to automatically regenerate zipped "resources"
# folders in each lesson folder. The zipped "resources" folder is used on the Petascale website
# lesson pages to allow users of the website to quickly download all resources for that lesson.
#
# To use: cd to '/petascale-semester-curriculum/code' directory in terminal / command line,
#     then run `./zip-resources`.
#     Allow script to run in about 10-15 seconds.
#
# Author: John Feshuk, johnf@shodor.org
# Created: 08/07/2020
#=================================================================================================

echo $(pwd) | grep petascale-semester-curriculum/code$ >/dev/null || {
  echo "You aren't in the proper place."
  echo "Navigate to '/petascale-semester-curriculum/code', then run './zip-resources'."
  exit 1
}

find ../units -maxdepth 3 -mindepth 3 -type d | while read line; do
  LESSON=$(basename "$line")
  if [ "${LESSON}" == 'in-progress' ]; then
    continue
  fi
  UNIT=$(basename $(dirname $(dirname "$line")))

  pushd .
  cd $line
  tar -czvf petascale-lesson-${UNIT}.${LESSON}-resources.tgz resources
  popd

done
