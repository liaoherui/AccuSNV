#!/bin/bash
sed -n '1~4s/^@/>/p;2~4p' ${1}
