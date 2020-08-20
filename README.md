<!--
*** Modified from othneildrew's README template
*** https://github.com/othneildrew/Best-README-Template/blob/master/README.md
-->


<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="http://shodor.org/petascale">
    <img src="images/BWPEPbanner.png" alt="Banner" width="801">
  </a>

  <h3 align="center">Blue Waters Petascale Semester Curriculum</h3>

  <p align="center">
    The Blue Waters Petascale Semester Curriculum is a collection of instructional materials for teaching parallel computing as applied to modeling and simulation. There are a total of 11 units with 6–12 lessons in each unit. Each lesson is designed to last 25 minutes of in-class time. The materials can be browsed and searched on <a href="http://shodor.org/petascale/materials/semester-curriculum">Shodor's Petascale site.</a>
    <br />
  </p>
</p>



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Using the Materials](#using-the-materials)
* [Contributing](#contributing)
* [Curriculum Developers](#curriculum-developers)
* [License](#license)
* [Contact](#contact)



<!-- ABOUT THE PROJECT -->
## About The Project

<a href="http://shodor.org">Shodor</a> has coordinated an effort to prepare faculty and professional staff to teach applied parallel modeling and simulation by incorporating the materials and lessons learned from the successful <a href="http://shodor.org/petascale/materials/institute/">Blue Waters Intern Petascale Institute</a> that has been conducted multiple times over the last ten years.

What has set the Petascale Institute apart from other efforts is that it starts with <a href="http://shodor.org/petascale/materials/modules"> working models solving problems in real science</a> that cover the continuum of multi-core/many-core technologies as exemplified by the Blue Waters project (such as OpenMP, MPI, OpenACC, CUDA, and hybrid). Instead of teaching these from an abstract, theoretical point, we have exploited the basic motifs used in real scientific code to teach undergraduate and graduate students the basics of applied parallel modeling and simulation.

The goal of the Petascale Semester Curriculum effort was to update the Petascale Institute materials to become more useful for preparing undergraduate and graduate students, as well as to help faculty and professional staff to integrate the materials into their own courses, student programs, REUs, workshops, and institutes.

This project is a component of the <a href="https://bluewaters.ncsa.illinois.edu/">Blue Waters</a> education initiatives at the <a href="https://illinois.edu/">University of Illinois</a>, funded by the <a href="https://www.nsf.gov/">National Science Foundation</a>.


<!-- HOW TO USE -->
## Using the Materials

These resources can be helpful both for *instructors* looking to integrate parallel computing materials into their courses and for *students* looking to learn more about parallel computing. The curriculum materials can be accesed on [Shodor's Petascale site](http://shodor.org/petascale/materials/semester-curriculum/units/).

For instructors:

* These materials are meant to provide example lessons that you can incorporate into your own course. We designed each lesson to take ~25 minutes of in class time so you can easily fit two topics into a 50 minute or 75 minute lecture.
* Additionally, Shodor plans to host a semesterly webinar for instructors looking to develop their parallel computing course. The webinars will be in August, September and December and more information on these webinars is TBD. If you are interested, please fill out the contact form at the bottom of any lesson page on the website.

For students:

* Watch our videos example lectures! Our developers put a lot of time into making these to help you understand the concepts more thoroughly.
* Go through our example code and example lesson plans/lecture notes.


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.


1. Fork the Project
2. Clone it to your local system
3. Make a new branch
4. Using the tables below, determine if your file is to be uploaded as a resource (indicated in the file type column)

**Unit Page**

Material                      | File Type           | File Name
:---------------------------- | :------------------ | :-----------------------------------------------------------------------
Unit Title / Description      | JSON                | "pageDetails.json"
Unit Learning Objectives      | HTML                | "objectives.html"
Unit Student Prerequisites    | HTML                | "studentPrereqs.html"
Unit Instructor Prerequisites | HTML                | "instructorPrereqs.html"
Unit Sample Assessment        | PDF / DOCX          | "petascale-unit-#-assessment.pdf" and "petascale-unit-#-assessment.docx"

<br />

**Lesson Page**

Material                                 | File Type           | File Name
:--------------------------------------- | :------------------ | :-----------------------------------------------------------------------
Lesson Title / Description / Developer   | JSON                | "pageDetails.json"
Lesson Learning Objectives               | HTML                | "objectives.html"
Lesson Student Prerequisites             | HTML                | "studentPrereqs.html"
Lesson Instructor Prerequisites          | HTML                | "instructorPrereqs.html"
Lesson Sample Assessment                 | PDF / DOCX          | "petascale-lesson-#.#-assessment.pdf" and "petascale-lesson-#.#-assessment.docx"
Curriculum Standards Addressed           | HTML                | "currStandards.html"
Materials Needed and System Requirements | HTML                | "systemReqs.html"
Presentation Slides                      | PDF / PPTX          | "petascale-lesson-#.#slides.pdf" and "petascale-lesson-#.#slides.pptx"
Presentation Video                       | JSON                | "presentationVideo.json"
Instructor Guide                         | PDF / DOCX          | "petascale-lesson-#.#-instructorGuide.pdf" and "petascale-lesson-#.#-instructorGuide.docx"
Code Example                             | GNU Zip             | "petascale-lesson-#.#-code.tar.gz"
Exercise Instructions for Students       | PDF / DOCX          | "petascale-lesson-#.#-exercise.pdf" and "petascale-lesson-#.#-exercise.docx"
References and Further Reading           | PDF / DOCX          | "petascale-lesson-#.#-references.pdf" and "petascale-lesson-#.#-references.docx"


* If you would like to update the description for a unit or lesson, simply edit the description line in the appropriate .json file

* If the file type *is not* preceded by "resource" in the table, you will have to upload your materials by editing the existing version of the HTML file that is in your branch. Make sure that the file you are editing is in the correct folder and is named correctly according to the table
    * You may also make changes to the curriculum material using your own text editor instead of directly editing the HTML. You can use a tool such as [Word 2 Clean HTML](https://word2cleanhtml.com/) for Word documents or [Docs to Markdown](https://gsuite.google.com/marketplace/app/docs_to_markdown/700168918607?pann=cwsdp&hl=en-US) for Google documents to convert text into HTML

* If the file type *is* preceded by "resource" in the table, please name your materials using the naming scheme provided and then upload your materials to the resources folder in the appropriate unit or lesson. This extra step allows the resources to be displayed directly on the webpage for each unit and lesson
5. Commit your changes
6. Click the Compare & pull request button
7. Click Create pull request to open a new pull request, and provide necessary details on what you’ve changed



<!-- CURRICULUM DEVELOPERS -->
## Curriculum Developers
* Beau Christ, Wofford College
* Colleen Heinemann, University of Illinois at Urbana-Champaign
* David P. Bunde, Knox College
* David A. Joiner, Kean University
* Hyacinthe Aboudja, Oklahoma City University
* Justin Oelgoetz, Austin Peay State University
* Marc Gagné, West Chester University
* Maria Pantoja, CalPoly San Luis Obispo
* Michael N Groves, California State University Fullerton
* Michael D. Shah, Northeastern University
* Nitin Sukhija, Slippery Rock University of Pennsylvania
* Paul F. Hemler, Hampden-Sydney College
* Peter J. Hawrylak, The University of Tulsa
* Roman Voronov, New Jersey Institute of Technology
* Sanish Rai, West Virginia University Institute of Technology
* Widodo Samyono, Jarvis Christian College

<!-- LICENSE -->
## License
License pending
<!-- Distributed under the MIT License. See `LICENSE` for more information. -->



<!-- CONTACT -->
## Contact

Via the Web
* Give us feedback in our [feedback form](http://shodor.org/petascale/feedback/).
* Please [contact us](http://shodor.org/petascale/contact/) if you have any questions, comments, or concerns.

Via Mail or Phone:
* The Shodor Education Foundation, Inc.
* 701 William Vickers Ave
     Durham, NC 27701
* Tel: (919) 530-1911
* Fax: (919) 530-1944


Project Link: [https://github.com/shodor-education/petascale-semester-curriculum](https://github.com/shodor-education/petascale-semester-curriculum)
