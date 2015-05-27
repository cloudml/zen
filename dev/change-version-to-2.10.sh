#!/usr/bin/env bash

#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Note that this will not necessarily work as intended with non-GNU sed (e.g. OS X)
# Copy from spark
BASEDIR=$(dirname $0)/..
find $BASEDIR -name 'pom.xml' | grep -v target \
  | xargs -I {} sed -i -e 's/\(artifactId.*\)_2.11/\1_2.10/g' {}

# Also update <scala.binary.version> in parent POM
sed -i -e '0,/<scala\.binary\.version>2.11</s//<scala.binary.version>2.10</' $BASEDIR/pom.xml
