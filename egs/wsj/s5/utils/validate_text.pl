#!/usr/bin/env perl
#===============================================================================
# Copyright 2017  (Author: Yenda Trmal <jtrmal@gmail.com>)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

# validation script for data/<dataset>/text
# to be called (preferably) from utils/validate_data_dir.sh
use strict;
use warnings;
use utf8;
use Fcntl qw< SEEK_SET >;

# this function reads the opened file (supplied as a first
# parameter) into an array of lines. For each
# line, it tests whether it's a valid utf-8 compatible
# line. If all lines are valid utf-8, it returns the lines
# decoded as utf-8, otherwise it assumes the file's encoding
# is one of those 1-byte encodings, such as ISO-8859-x
# or Windows CP-X.
# Please recall we do not really care about
# the actually encoding, we just need to
# make sure the length of the (decoded) string
# is correct (to make the output formatting looking right).
sub get_utf8_or_bytestream {
  use Encode qw(decode encode);
  my $is_utf_compatible = 1;
  my @unicode_lines;
  my @raw_lines;
  my $raw_text;
  my $lineno = 0;
  my $file = shift;

  while (<$file>) {
    $raw_text = $_;
    last unless $raw_text;
    if ($is_utf_compatible) {
      my $decoded_text = eval { decode("UTF-8", $raw_text, Encode::FB_CROAK) } ;
      $is_utf_compatible = $is_utf_compatible && defined($decoded_text);
      push @unicode_lines, $decoded_text;
    } else {
      #print STDERR "WARNING: the line $raw_text cannot be interpreted as UTF-8: $decoded_text\n";
      ;
    }
    push @raw_lines, $raw_text;
    $lineno += 1;
  }

  if (!$is_utf_compatible) {
    return (0, @raw_lines);
  } else {
    return (1, @unicode_lines);
  }
}

# check if the given unicode string contain unicode whitespaces
# other than the usual four: TAB, LF, CR and SPACE
sub validate_utf8_whitespaces {
  my $unicode_lines = shift;
  use feature 'unicode_strings';
  for (my $i = 0; $i < scalar @{$unicode_lines}; $i++) {
    my $current_line = $unicode_lines->[$i];
    # we replace TAB, LF, CR, and SPACE
    # this is to simplify the test
    $current_line =~ s/[\x{0009}\x{000a}\x{000d}\x{0020}]/./g;
    if ($current_line =~/\s/) {
      return 1;
    }
  }
  return 0;
}

# checks if the text in the file (supplied as the argument) is utf-8 compatible
# if yes, checks if it contains only allowed whitespaces. If no, then does not
# do anything. The function seeks to the original position in the file after
# reading the text.
sub check_allowed_whitespace {
  my $file = shift;
  my $pos = tell($file);
  (my $is_utf, my @lines) = get_utf8_or_bytestream($file);
  seek($file, $pos, SEEK_SET);
  if ($is_utf) {
    my $has_invalid_whitespaces = validate_utf8_whitespaces(\@lines);
    print "--> text seems to be UTF-8 or ASCII, checking whitespaces\n";
    if ($has_invalid_whitespaces) {
      print "--> ERROR: the text containes disallowed UTF-8 whitespace character(s)\n";
      return 0;
    } else {
      print "--> text contains only allowed whitespaces\n";
    }
  } else {
    print "--> text doesn't seem to be UTF-8 or ASCII, won't check whitespaces\n";
  }
  return 1;
}
if(@ARGV != 1) {
  die "Usage: validate_text.pl <text-file>\n" .
      "e.g.: validate_text.pl data/train/text\n";
}

my $text = shift @ARGV;

# Checking optional_silence.txt -------------------------------
print "Checking $text ...\n";
if(-z "$text") {
  print "--> ERROR: $text is empty or not exists\n";
  exit 1;
}

if(!open(FILE, "<$text")) {
  print "--> ERROR: fail to open $text\n";
  exit 1;
}

print "--> reading $text\n";
check_allowed_whitespace(\*FILE) or exit 1;
close(FILE);


