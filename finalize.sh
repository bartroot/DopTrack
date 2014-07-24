#!/bin/bash -u

PROJECT="`cat .projectname`.pdf"
APPENDIX="D14_ScientificArticleFinal.pdf A6_final.pdf"
FINAL="`cat .projectname`.final.pdf"

pdftk $PROJECT $APPENDIX output $FINAL
