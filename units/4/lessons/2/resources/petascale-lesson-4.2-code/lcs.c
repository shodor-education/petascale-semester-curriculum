/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 4: OpenMP
 * Lesson 2: Longest Common Subsequence
 * File: lcs.c
 * Developed by Paul F. Hemler for the Shodor Education Foundation, Inc.
 *
 * Copyright (c) 2020 The Shodor Education Foundation, Inc.
 *
 * Browse and search the full curriculum at
 * <http://shodor.org/petascale/materials/semester-curriculum>.
 *
 * We welcome your improvements! You can submit your proposed changes to this
 * material and the rest of the curriculum in our GitHub repository at
 * <https://github.com/shodor-education/petascale-semester-curriculum>.
 *
 * We want to hear from you! Please let us know your experiences using this
 * material by sending email to petascale@shodor.org
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

/*
Program finds the longest common subsequence between a text
string and a pattern string.  The dynamic programming approach
is implemented.  There are three functions to fill the 
cost table.  The first one is row by row.  Unfortunately, 
there is a data dependency that makes this approach impossible
to make parallel with OpenMP.  A second function fills the table
diagonally to avoid the data dependency.  This function is then
made parallel with OpenMP.  All three functions are timed as
they fill up an empty table.  The times to fill the table are
then reported.  The program was executed with different numbers
of threads to see how that affects performance.

There are two data files associated with this program
*/

// Standard includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <omp.h>

// The number of threads OpenMP should spawn if not specified on
// the command line
int nThreads = 4;


// My own max function
int myMax(int a, int b) {
	return a >= b ? a : b;
}

// Table has one extra row and column
// The first row and column is assumed to be initialized with 0's
// It is assumed the number of rows is less than the number of columns
// The text string goes across the top and the pattern goes along
// the right side of the table
void fillTableByRows(int** table, char* pattern, char* text)
{
    // The number of rows and columns
    int nr = strlen(pattern);
    int nc = strlen(text);

    // Fill the table one row at a time
    for (int r = 1; r <= nr; ++r)
		for (int c = 1; c <= nc; ++c)
		{
			if (pattern[r - 1] == text[c - 1])
				table[r][c] = table[r - 1][c - 1] + 1;
			else
				table[r][c] = myMax(table[r - 1][c], table[r][c - 1]);
		}
}


// Table has one extra row and column
// The first row and column is assumed to be initialized with 0's
void fillTableByDiagonals(int** table, char* pattern, char* text)
{
    // assumes nr <= nc
    int nr = strlen(pattern);
    int nc = strlen(text);
	
    int diag, diagSize, k;

    // Fill the table one diagonal at a time
    for (diag = 1; diag <= nr + nc - 1; ++diag) {
		diagSize = diag;
		if (diag >= nr)
			diagSize = nr;
		if (diag >= nc)
			diagSize = nr - (diag - nc);

		// Move along the diagonal starting on the top row and
		// ending on either the left or bottom edge
		for (k = 0; k < diagSize; ++k) {
			int r = diag - k;
			int c = k + 1;
			if (diag > nr) {
				r = nr - k;
				c = diag - (nr - 1) + k;
			}

			// Either the corresponding text and pattern match or
			// they do not match.  Either way, we know what to put
			// in the table
			if (pattern[r - 1] == text[c - 1])
				table[r][c] = 1 + table[r - 1][c - 1];
			else
				table[r][c] = myMax(table[r - 1][c], table[r][c - 1]);
		}
    }
}


// OpenMP parallel version of the function above
void fillTableByDiagonalsP(int** table, char* pattern, char* text)
{
    // assumes nr <= nc
    int nr = strlen(pattern);
    int nc = strlen(text);
	
    int diag, diagSize, k;

    // Fill the table one diagonal at a time
    for (diag = 1; diag <= nr + nc - 1; ++diag) {
		diagSize = diag;
		if (diag >= nr)
			diagSize = nr;
		if (diag >= nc)
			diagSize = nr - (diag - nc);

		// Move along the diagonal starting on the top row and
		// ending on either the left or bottom edge

#pragma omp parallel for num_threads(nThreads) default(none) shared(table, diag, diagSize, nr, text, pattern)
		for (k = 0; k < diagSize; ++k) {
			int r = diag - k;
			int c = k + 1;
			if (diag > nr) {
				r = nr - k;
				c = diag - (nr - 1) + k;
			}

			// Either the corresponding text and pattern match or
			// they do not match.  Either way, we know what to put
			// in the table
			if (pattern[r - 1] == text[c - 1])
				table[r][c] = 1 + table[r - 1][c - 1];
			else
				table[r][c] = myMax(table[r - 1][c], table[r][c - 1]);
		}
    }
}

// Traverse the table to determine the longest common substring
int findMatch(int** table, char* pattern, char* text, char* match)
{
    int r = strlen(pattern);
    int c = strlen(text);
    int length = table[r][c];

    //Move backwards up the table to find the matching subsequence
    while (r > 0 && c > 0 && length > 0)
    {
		if (pattern[r - 1] == text[c - 1])
		{
			match[length - 1] = pattern[r - 1];
			r--;
			c--;
			length--;
		}
	else if (table[r][c - 1] > table[r - 1][c])
	    c--;
	else
	    r--;
    }
    return c;
}


// The main driver program.  Two command line arguments are optional.
// The first argument is the number of threads that should be spawned
// and the second option is for a text file containing the strings
// to be matched.  If a text file is not specified on the command
// line there are some optional smaller text and patterns strings
// used for testing purposes.
// The file format contains the length of the text, the text as
// one long string of characters, the length of the pattern,
// followed by the pattern as one long string of characters.
// The text is made to be longer than the pattern.  The text goes
// across the table and the pattern goes along the left side.  The
// table has one extra row and column filled with zeros, which
// assists in filling the table.  Once the table is filled it can
// be traversed backward to find the longest common substring.
// Three functions for filling the table are invoked and timed.
int main(int argc, char* argv[])
{
    //Declare variables
    unsigned int lenText, lenPattern;
    char *strText, *strPattern;
    char* match;
    int** table;

    // Check for the first command line argument, the number of threads
    if (argc >= 2)
    {
		nThreads = strtol(argv[1], NULL, 10);
		if (nThreads < 1) { nThreads = 4; }
    }

    // Check for two command line arguments and read the file
    if (argc == 3)
    {
		// Open the file
		FILE *fptr = fopen(argv[2], "r");
		if (!fptr)
		{
			fprintf(stderr, "ERROR: Could NOT open the file: %s\n", argv[2]);
			exit(-1);
		}

		// Read the number of characters in the text and then the text
		int status = fscanf(fptr, "%d", &lenText);
		strText = (char*)malloc(lenText * sizeof(char));
		if (fscanf(fptr, "%s", strText) != 1) 
		{
			fprintf(stderr, "ERROR: failed to read the text string\n");
			exit(-1);
		}

		// Read the number of characters in the pattern and then the pattern
		status = fscanf(fptr, "%d", &lenPattern);
		strPattern = (char*)malloc(lenPattern * sizeof(char));
		if (fscanf(fptr, "%s", strPattern) != 1)
		{
			fprintf(stderr, "ERROR: failed to read the pattern string\n");
			exit(-1);
		}
		fclose(fptr);
    }
    // If not command line argument is specified, use one of these text/patterns
    else {
		// char* theText = "CGCTTAAAATTGGGAGTGGTTGATGCTCTATACTCCATTTGGTTTTTTCGTGCATCACCGCGATAGGCTGACAAGGGTTTAACATTGAATAGCAAGGCACTTCCGGTCTCAATGAAGGGCCGGGAAAG";
		// char* thePattern = "ATATATGTATGGTATAAAAAAATTAAGAAATCAATAAAAAAACTTTTTCCTTACTTATTAAAAAATAAATACAGAATAATAATTACTATATTAATATGTGAAACATATTTAATAATTATTTAAAGTA";
		// char* theText = "ABCD";
		// char* thePattern = "FD";
		char* theText = "GCTCAGC";		// correct answer is 4 (GTAC)
		char* thePattern = "AGGTAC";
		strText = strdup(theText);
		strPattern = strdup(thePattern);

		lenText = strlen(strText);
		lenPattern = strlen(strPattern);
    }

    // Switch so there are more columns than rows.  The text should be
    // longer than the pattern.  Caracters of the text go across the top
    // and characters of the pattern go along the left side.  Fill the
    // table one row at a time.  Move left when determining the matching
    // characters from the filled table
    if (lenPattern > lenText)
    {
		char* tmpChar = strText;
		strText = strPattern;
		strPattern = tmpChar;
		lenText = strlen(strText);
		lenPattern = strlen(strPattern);
    }

    // Make sure all characters in the text and pattern are upper case
    for (int i = 0; i < lenText; ++i)
		strText[i] = (char)toupper(strText[i]);
    for (int i = 0; i < lenPattern; ++i)
		strPattern[i] = (char)toupper(strPattern[i]);

    // There cannot be more characters in the matching string 
    // then there are in the pattern
    match = (char*)malloc((lenPattern + 1) * sizeof(char));

    // Initialize the matching string
    memset(match, 0, (lenPattern + 1) * sizeof(char));

    // Allocate memory for the table, lenPattern X lenText
    // Also resolve the row pointers
    table = (int**)malloc((lenPattern + 1) * sizeof(int*));
    table[0] = (int*)malloc((lenText + 1) * (lenPattern + 1) * sizeof(int));
    for (int i = 1; i <= lenPattern; i++)
		table[i] = &table[0][i * (lenText + 1)];

    // Initialize the table to all zeros
    memset(table[0], 0, (lenText + 1) * (lenPattern + 1) * sizeof(int));

    // Measure and display the time to fill the table row by row
    double time1, time2;
    time1 = omp_get_wtime();
    fillTableByRows(table, strPattern, strText);
    time2 = omp_get_wtime();
    printf ("serial time rows: %.3lf\n", time2 - time1);
    printf("Number of characters that match: %d\n", table[lenPattern][lenText]);

    // Reset the table
    memset(table[0], 0, (lenText + 1) * (lenPattern + 1) * sizeof(int));

    // Measure and display the time to fill the table by diagonals
    time1 = omp_get_wtime();
    fillTableByDiagonals(table, strPattern, strText);
    time2 = omp_get_wtime();
    printf ("Serial time diagonals: %.3lf\n", time2 - time1);
    printf("Number of characters that match: %d\n", table[lenPattern][lenText]);

    // Reset the table
    memset(table[0], 0, (lenText + 1) * (lenPattern + 1) * sizeof(int));

    // Measure and display the time to fill the table by diagonals in parallel
    time1 = omp_get_wtime();
    fillTableByDiagonalsP(table, strPattern, strText);
    time2 = omp_get_wtime();
    printf ("Parallel time: %.3lf  number of threads: %d\n", time2 - time1, nThreads);
    printf("Number of characters that match: %d\n", table[lenPattern][lenText]);

    // Given the table is filled, find the matching string
    int matchStartIndex = findMatch(table, strPattern, strText, match);
    printf("The match starts in the text at index: %d\n", matchStartIndex);

    // Print the Longest Common Substring if you like
//    printf("The match: %s\n", match);

    // Be kind rewind
    free(table[0]);
    free(table);
    free(match);
    free(strText);
    free(strPattern);

    // All done
    return 0;
}

