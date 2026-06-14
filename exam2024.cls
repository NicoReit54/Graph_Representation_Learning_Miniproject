\ProvidesClass{exam2024}[2015/10/28]

\LoadClass[11pt]{article}

%%%%%%%%%% Page layout

\setlength\paperheight{297mm}
\setlength\paperwidth{210mm}
\setlength\textheight{\paperheight}
\addtolength\textheight{-70mm}
\setlength\topmargin{-10mm}
\setlength\textwidth{\paperwidth}
\addtolength\textwidth{-50mm}
\setlength\oddsidemargin{0pt}
\setlength\evensidemargin {0pt}
\advance\footskip by\normalbaselineskip

%%%%%%%%%% Page styles

% Kill LaTeX's use of marks
\def\markboth#1#2{}
\def\markright#1{}

\mark{\noexpand\@turning}

\def\@turning{\ifodd\c@page TURN OVER\fi}

% During the endgame, we want to label the very last page: we delay
% doing so while there are still deferred floats.
\def\@lastturn{\ifx\@deferlist\@empty LAST PAGE\else\@turning\fi}

\newif\ifjustparent \justparentfalse

\def\justparentfooter{\justparenttrue}

% Extract the main code from the list of exam codes
\def\@footercodes{\ifx\@codes\undefined \else 
  \ifjustparent {\tt\expandafter\justp@rent\@codes\\\done}
  \else \vbox{\small\tt \baselineskip=1em \def\\{\cr}%
    \halign{##\hfil\cr\@codes\crcr}}\fi\fi}
\def\justp@rent#1\\#2\done{#1}

% Page style for question pages
\def\ps@exam{%
 \def\@oddhead{}%
 \def\@evenhead{}%
 \def\@oddfoot{%
    \rlap{\@footercodes}\hfil\rm --\ \thepage\ --\hfil\llap{\bf\botmark}}
 \let\@evenfoot=\@oddfoot}

% Page style for front cover
\def\ps@front{%
 \def\@oddhead{}%
 \def\@evenhead{}%
 \def\@oddfoot{\hfil\rm --\ \thepage\ --\hfil}%
 \let\@evenfoot=\@oddfoot}

% Page style for model answers
\def\ps@answers{%
 \def\@oddhead{}%
 \def\@evenhead{}%
 \def\@oddfoot{\hfil\rm --\ \thepage\ --\hfil}%
 \let\@evenfoot=\@oddfoot}


%%%%%%%%%% Numbering of questions and parts

\newcounter{question}

% \if@inquestion controls whether the 'note' environment refers to a
% particular question
\newif\if@inquestion \@inquestionfalse

% Specify title of paper for drafts
\def\paper#1{\if@wantfront\else \section*{#1}\fi}

% Format a question
\def\question{\par
  \@inquestiontrue
  \refstepcounter{question}
  \parindent=0pt \parskip=\smallskipamount \@parhack
  \setcounter{enumi}{0}
  \subsection*{Question \arabic{question}}
  \@ifnextchar[\@question\relax}
\def\@question[#1]{\def\mks@q{#1}\@markstrue}
\def\@finishquestion{\ifx\mks@q\undefined\else\@marks\mks@q\fi}
\def\endquestion{\@finishquestion \@finishanswer1\global\@ansdepth=0}

% Format a model answer: this appears in the file <jobname>.ans that is
% written as the questions are processed.
\def\savedanswer#1{\par
  \parindent=0pt \parskip=\smallskipamount
  \setcounter{enumi}{0}
  \subsection*{Answer to question #1}}

% Format a note as part of the model answers.  Bad formatting of the
% answer file may lead to a blank line before the note starts, so we
% use \everypar to attach the label to it.
\newif\if@newnote \@newnotefalse

\def\ansnote{\par \@newnotetrue
  \everypar={\global\@nobreakfalse\everypar={}%
    \if@newnote\global\@newnotefalse {\it Note\/}: \ \fi}}
\def\endansnote{\par}

\def\thequestion{\arabic{question}}

% We hijack 'enumerate' for numbering parts of questions.  It would be
% simple but tedious to avoid this.
\let\s@vedenumerate=\enumerate
\def\enumerate{\errmessage{Please don't use 'enumerate' -- use 'part' instead}}

% Each 'part' is a separate list environment.  This means we get the same
% spacing between parts as we get before the first and after the last,
% but that is not really such a bad thing.

% We keep the marks for each nesting level in separate
% variables \mks@enumi, \mks@enumii (etc.) so
% that we can reproduce the marks in the model answer

\newif\if@marks \@marksfalse

% Format a part of a question
\def\part{\s@vedenumerate\reset@counter\@parhack
  \let\answer=\@partanswer \item\p@rt}
\def\p@rt{\@ifnextchar[\@part\relax}
\def\@part[#1]{\if@marks
    \errmessage{An ancestor of this part also has a marks allocation}\fi
  \expandafter\def\csname mks@\@enumctr\endcsname{#1}\@markstrue}
\def\@finishpart{\do@marks\@marks\@enumctr\endlist}
\def\endpart{\@finishpart \@finishanswer\@enumdepth}

% Format a part of a model answer: \begin{anspart}{label}[marks]
\def\anspart#1{\s@vedenumerate
  \@parhack
  \item[#1]%
  \p@rt}
\let\endanspart=\@finishpart

% Call a function on the marks, if there are any for this part
\def\do@marks#1#2{\expandafter\ifx\csname mks@#2\endcsname\relax
  \else\expandafter#1\csname mks@#2\endcsname\fi}

% When we start an enumerate, we reset the counter for the next level in.
% Luckily we need i -> ii and maybe ii -> iii but not iii -> iv; so
% the cave man's algorithm for incrementing is good enough
\def\reset@counter{\setcounter{\@enumctr i}{0}}

% Modify the defn from latex.ltx -- don't reset counter in mid-question
\def\usecounter#1{\@nmbrlisttrue\def\@listctr{#1}}

% Unlike the article style, crossrefs produce (a)(i) not ai.
\def\theenumi{(\@alph\c@enumi)}
\def\labelenumi{\theenumi}
\def\theenumii{(\@roman\c@enumii)}
\def\labelenumii{\theenumii}


%%%%%%%%%% Marks at the end of each part

% We use the \@marks macro to attach the marks allocation to the end of
% each part of the question.  This can be done via the optional
% argument to the 'part' environment, which also saves the marks for
% the model answer.  But experience shows that it's tempting to add a
% blank line before \answer, and this messes things up by finishing
% the last paragraph of the question before \@marks has chance to
% attach the marks.  A solution is to override the \par operation so
% that a \par before \answer or \end is ignored; we must do this
% delicately, because LaTeX itself is also manipulating the definition
% of \par around lists.  What's written below seems to work most of
% the time (!), but there was a real rat's nest here even before we arrived.

% LaTeX contains the following definitions:
%
% \let\@@par=\par
%
% \def\@setpar#1{\def\par{#1}\def\@par{#1}}
%
% \def\@par{\let\par=\@@par \par}
%
% \def\@restorepar{\def\par{\@par}}
%
% The LaTeX conventions are as follows:
%
% 			\par	      	\@par
%                       -----------------------------
%   Default state: 	= \@@par      	default value (see above)
%   Resetting: 		m: \@par      	default value
%   After resetpar:	m: \@par	macro
%   Long term:		macro      	same macro
%   Short term:		macro	      	???
%
% The default value of \@par is functionally the same as \@@par, but has
% the side-effect, beneficial for efficiency, of making \par invoke the
% primitive \@@par directly.  I don't see why \@restorepar is not defined
% as \let\par=\@par, but that probably has to do with efficiency too.
%
% Our conventions are simpler:
%
%   Default:		= \hack@par	\@@par
%   Long term:		= \hack@par	macro
%   Short term:		macro		???

\def\@parhack{%
  \ifx\@setpar\my@setpar\else % Do nothing if already hacked
    % Set \par unless it already has a short-term value
    \ifx\par\@par \let\par=\hack@par
    \else\ifx\par\mac@par \let\par=\hack@par \fi
    \fi
    % Establish special versions of \@setpar and \@restorepar
    \let\@setpar=\my@setpar \let\@restorepar=\my@restorepar
  \fi}

\def\mac@par{\@par}

%% I'm not sure what's best here, so I'm being conservative.  But I think
%% things would work the same and be simpler if the following changes
%% were made:
%\let\@par=\@@par % Remove the tricky side-effectful definition
%\def\@restorepar{\let\par=\@par} % Restore the value of \par directly
%% Also, remove the line from \@parhack beginning \else\ifx ...,
%% and remove the definition of \mac@par.

\def\my@setpar#1{\let\par=\hack@par \def\@par{#1}}
\def\my@restorepar{\let\par=\hack@par}

% And our fancy version of \par: it does nothing if it is followed by
% an \answer, an \end, or another \par, so that a subsequent
% \@marks can be attached properly, and otherwise calls \@par.
% To cope with the old-style interface, it also suppresses \@par before
% a call to \@marks.
\def\hack@par{\futurelet\hp@tmp\@hack@par}
\def\@hack@par{\ifx\hp@tmp\answer\else\ifx\hp@tmp\end
  \else\ifx\hp@tmp\par\else\ifx\hp@tmp\@marks\else\@par\fi\fi\fi\fi}

% Here's how to set the marks at the end of a paragraph, with a
% long sequence of penalties and glue.
\def\@marks#1{\leavevmode\unskip\nobreak\hfill\penalty100\null\nobreak\hfill
  \quad\hbox{\sf (#1 \edef\next{#1}\ifx\next\@one mark\else marks\fi)}\par}

\def\@one{1} 


%%%%%%%%%% Saving answers

% The answers to each part, or optionally for a whole question at
% once, are saved on a file to be read back later.  The macros here
% scan the answer as a macro argument, which is consistent with
% the way LaTeX treats e.g. verbatim text.  An earlier solution used a
% loop as suggested in the TeXbook to copy the text line by line.
% That solution is a bit more tricky to get right.

% The contents of the answer file tend to be a bit messy, with unruly
% alignment and stray blank lines.  It doesn't really matter much, and
% is not worth tidying up for present purposes.

% We keep track of the current nesting level for model answers as the
% counter \@ansdepth, and use this to open and close parts as needed
% to mirror the structure of the question in the model answer.

\newwrite\@ansout
\newif\if@ansopen \@ansopenfalse
\newcount\@ansdepth

% Write on the answer file
\def\@writeans{\if@ansopen\else
    \immediate\openout\@ansout=\jobname.ans\global\@ansopentrue\fi
  \immediate\write\@ansout}

% Ditto without expansion
\def\@@writeans#1{\toks0={#1}\@writeans{\the\toks0}}

% If any answers have been written, read them back for printing
\def\do@answers{\if@ansopen
    \immediate\closeout\@ansout
    \immediate\write-1{anspage=\the\count0}
    \mark{}
    \pagestyle{answers}
    \pagenumbering{roman}
    \count1=1 % This gives page numbers 1.1, 2.1, ... for use with dviselect
    \section*{\ifx\@papername\undefined\else\@papername:\ \hfil\break \fi
        Model answers\ifx\term\undefined\else\ for \term\fi}
    \let\note=\ansnote \let\endnote=\endansnote 
        % In case any notes are embedded in the answer file.
    \input\jobname.ans\relax
  \fi}

% Save answer from a whole question
\def\answer{\if@ansinline
    \par\small {\it Answer\/}: \
    \setcounter{enumi}0
    \let\next=\relax
  \else
    \@finishanswer1\@finishquestion\endgroup
    \ifnum\@ansdepth=0 \@writeans{\string\savedanswer{\thequestion}}\fi
    \global\@ansdepth=0
    \begingroup\@copyit\let\next=\write@answer
  \fi
  \next}
\def\endanswer{\par}

% Save answer from one part (the 'part' envt redefines \answer to this)
\def\@partanswer{\if@ansinline
    \par\small {\it Answer\/}: \
    \let\next=\relax
  \else
    \@newanswer\@finishpart\endgroup%
    \begingroup\@copyit\let\next=\write@part
  \fi
  \next}

% Prepare for writing a new answer by shifting levels up or down
\def\@newanswer{%
  \@finishanswer\@enumdepth
  \ifnum\@ansdepth=0           	% Start the question if necessary
    \@writeans{\string\savedanswer{\thequestion}}%
    \global\@ansdepth=1
  \fi
  \loop				% Start parts up to this level
    \edef\@ansctr{enum\romannumeral\the\@ansdepth}%
    \@writeans{\string\begin{anspart}%
        {\csname the\@ansctr\endcsname}\do@marks\sqbrack\@ansctr}
  \ifnum\@ansdepth<\@enumdepth
    \global\advance\@ansdepth by1
  \repeat}

% Put marks in square brackets as optional argument to 'anspart'
\def\sqbrack#1{[#1]}

% Close nested answers down to a specified level
\def\@finishanswer#1{%
  \loop\ifnum\@ansdepth>#1
    \@writeans{\string\end{anspart}}%
    \global\advance\@ansdepth by-1
  \repeat}

% Write a note to the model answers
\def\note{\if@ansinline
    \par\small {\it Note\/}: \
    \let\next=\relax
  \else
    \if@inquestion\ifnum\@ansdepth=0   	% Start the question if necessary
      \@writeans{\string\savedanswer{\thequestion}}%
      \global\@ansdepth=1
    \fi\fi
    \@writeans{\string\begin{ansnote}}
    \@copyit\let\next=\write@note
  \fi
  \next}
\def\endnote{\par}

{\obeylines
  \gdef\@copyit{\let\do=\@makeother\dospecials\obeylines%
    \let^^M=\relax\newlinechar=`\^^M}}

{\catcode`\|=0 \catcode`\[=1 \catcode`\]=2
  \catcode`\{=12 \catcode`\}=12 \catcode`\\=12 
  |gdef|write@answer#1\end{question}[|@@writeans[#1]|endgroup]
  |gdef|write@part#1\end{part}[|@@writeans[#1\end{anspart}]|endgroup]
  |gdef|write@note#1\end{note}[|@@writeans[#1\end{ansnote}]|endgroup]]


%%%%%%%%%% Data for front page

% Data for the front pages of all papers lives in the file front.tex
% A typical record is as follows:
%
%   \front{sample}{csmods,cs=CS4,mcs,ecs=ECS4}
%   {Subject Oriented Programming}
%   {Friday 8th June, 9:30\am\ -- 12:30\pm}
%   {\corerubric}
%   {DCOM 1234\\DMCN 1234}
%
% The keys csmods, cs, etc. are defined by the \dcl... macros below to
% call up appropriate elements on the cover page.  Pieces of
% boilerplate like \corerubric are also defined in front.tex in terms
% of macros like \rubric that are defined here.  All the data is
% stored in various macros under the key 'sample', which will be the
% value of \jobname when the file sample.tex is being processed.

\newtoks\allfps \allfps{}

\def\@addtofplist#1{\expandafter\global\expandafter\allfps
  \expandafter{\the\allfps#1}}

% Store front page data for a paper.
\def\front#1#2#3#4#5#6{% jobname schools title date rubric codes
  \expandafter\gdef\csname fp@#1\endcsname
    {\frontpage{#2}{#3}{#4}{#5}{#6}}%
  \expandafter\@addtofplist\csname fp@#1\endcsname}

% Conditions controlling "FIRST PUBLIC EXAMINATION" etc.
\newif\if@firstpub \@firstpubfalse
\newif\if@secondpub \@secondpubfalse
\newif\if@masters \@mastersfalse

% List of examinations "Honour Moderations in Computer Science" etc.
\newtoks\allexams \allexams={}

% Declare an examination
\def\dcl@exam#1#2#3{% option title pubtrue
  \expandafter\newif\csname if@#1\endcsname 
  \expandafter\def\csname tit@#1\endcsname{#2}
  \expandafter\def\csname pub@#1\endcsname{#3}
  \expandafter\def\csname ppr@#1\endcsname{\csname @#1paper\endcsname}
  \expandafter\allexams\expandafter{\the\allexams\do{#1}}}

% The order here is the order the exams appear on the front page, and 
% the order they are specified in the front page data has no effect.

% Keep these forever so that we can format old papers
\def\dcl@mods#1#2{\dcl@exam{#1}{Honour Moderations in #2}\@firstpubtrue}
\dcl@mods{csmods}{Computer Science}
\dcl@mods{mcsmods}{Mathematics and Computer Science} 
  \let\ppr@mcsmods=\ppr@csmods % Same paper numbers for CS and M&CS mods

\def\dcl@prelims#1#2{\dcl@exam{#1}{Preliminary Examination in #2}\@firstpubtrue}
\dcl@prelims{csprelims}{Computer Science}
\dcl@prelims{mcsprelims}{Mathematics and Computer Science}
  \let\ppr@mcsprelims=\ppr@csprelims
\dcl@prelims{cspprelims}{Computer Science and Philosophy}
  \let\ppr@cspprelims=\ppr@csprelims

\def\dcl@school#1#2{\dcl@exam{#1}{Honour School of #2}\@secondpubtrue}
\dcl@school{cs}{Computer Science}
\dcl@school{mcs}{Mathematics and Computer Science} 
\dcl@school{csp}{Computer Science and Philosophy}
\dcl@school{ecs}{Engineering and Computing Science}
\dcl@school{maths}{Mathematics}
\dcl@school{mphil}{Mathematics and Philosophy}

\def\dcl@masters#1#2{\dcl@exam{#1}{#2}\@masterstrue}
\dcl@masters{msc}{\if@secondpub Master of Science in \fi Computer Science}
  % We say 'Master of Science in' for shared papers

% Process a school spec like cs=CS4
\def\do@school#1=#2=#3\done{%
  \csname @#1true\endcsname 
  \csname pub@#1\endcsname
  \def\next{#2}%
  \ifx\next\empty\else
    \expandafter\def\csname @#1paper\endcsname{#2}\fi}

% Invoking \do@school X==\done with 'cs' for X gives #1='cs', #2='', #3=''.
% Using 'cs=CS4' for X gives #1='cs', #2='CS4', #3='='.

% Process a list of schools specs: the order they appear doesn't matter
\def\do@schools#1,#2\done{\do@school#1==\done
  \def\next{#2}\ifx\next\empty\else\do@schools#2\done\fi}

% Say \takehome before \frontpage for a take-home exam (no 'Do not turn over')
\newif\if@kickoff \@kickofftrue
\def\takehome{\@kickofffalse}


%%%%%%%%%% Formatting the front page

% Embedded data for front page, for use in ad-hoc papers
\def\frontpage#1#2#3#4#5{% schools title date rubric codes
  \fp@reset
  \do@schools#1,\done
  \gdef\@papername{#2}%
  \gdef\@codes{#5}%
  \@makefrontpage{#2}{#3}{#4}}

% Reset the variables governing the front page: needed for allfps.tex
\def\fp@reset{\@firstpubfalse\@secondpubfalse\@mastersfalse
  \def\do##1{\csname @##1false\endcsname
    \expandafter\def\csname @##1paper\endcsname{}}
  \the\allexams}

% Why is \centerline deleted by LaTeX?
\def\@centerline#1{\hbox to\hsize{\hfil #1\hfil}}

% Format e.g. Honour School of Computer Science: Paper CS2
\def\fmt@exam#1{
  \csname if@#1\endcsname
    \edef\next{\csname ppr@#1\endcsname}%
    \@centerline{\large \bf \csname tit@#1\endcsname
% Schools don't seem to want this any more ...
      \ifx\next\empty\else : Paper \next\fi
    }\medskip\fi}

% Format the front page
\def\@makefrontpage#1#2#3{% title date rubric
  \clearpage
  \thispagestyle{front}
  \vtop to0pt{\baselineskip=1em \def\\{\hfil\break}%
    \noindent\tt\@codes\par\vss}
  \vskip 72pt
  \if@firstpub\@centerline{\Large\bf FIRST PUBLIC EXAMINATION}\medskip\fi
  \if@secondpub\@centerline{\Large\bf SECOND PUBLIC EXAMINATION}\medskip\fi
  \if@masters\@centerline{\Large\bf DEGREE OF MASTER OF SCIENCE}\medskip\fi
  \medskip
  % Honour School of blah blah blah
  {\let\do=\fmt@exam\the\allexams}
  \medskip\bigskip\bigskip
  \hrule
  \bigskip\bigskip
  % Title of paper
  {\Large\bf\centering\parskip=0pt #1\par}
  \bigskip\bigskip
  \hrule
  \bigskip\bigskip\bigskip
  % Trinity Term 20xx
  \@centerline{\large\bf\expandafter\uppercase\expandafter{\term}}
  \medskip
  % Date and time
  \@centerline{\large\bf \def\am{\thinspace am}%
    \def\pm{\thinspace pm}#2}
  #3
  \vfill\vfill
  \if@kickoff
    \@centerline{Do {\bf not} turn over until told that you may do so.}
  \fi
  \newpage}
  
\def\@rubric#1{\bigskip\bigskip{\centering #1\par}}
\def\rubric#1{\@rubric{\it #1}}
\def\rules#1{\@rubric{\rm #1}}
\def\warning#1{\@rubric{\bf #1}}

%%%%%%%%%%% Initialization and options

\newif\if@standalone \@standalonefalse
\newif\if@wantanswers \@wantanswerstrue
\newif\if@answers \@answersfalse
\newif\if@ansinline \@ansinlinefalse
\newif\if@wantfront \@wantfrontfalse

% Switches for use in globopt.tex
\def\modelanswers{\@answerstrue}
\def\frontpages{\@wantfronttrue\input front}

\DeclareOption{standalone}{\@standalonetrue}
\DeclareOption{noanswers}{\@wantanswersfalse}
\DeclareOption{inlineanswers}{\@ansinlinetrue}

\ProcessOptions

\AtBeginDocument{%
  \if@standalone
    \if@wantanswers\@answerstrue\fi
  \else
    \InputIfFileExists{globopt.tex}{}{\if@wantanswers\@answerstrue\fi}%
  \fi}

% In order to allow certain graphics packages to inject Postscript
% in the right place, we mustn't do the front page via \AtBeginDocument.
% So we're back to the old hack of hijacking \document.

\let\@olddocument=\document
\def\document{\@olddocument\@dofront}

% \paperkey is defined specially in answers.tex; here is the default
\ifx\paperkey\undefined \edef\paperkey{\jobname}\fi

\def\@dofront{\if@wantfront
    \expandafter\ifx\csname fp@\paperkey\endcsname\relax
      \PackageWarning{exam2}{No front page data for paper \paperkey}%
    \else \csname fp@\paperkey\endcsname \fi
  \fi}

\AtEndDocument{%
  \vskip-\lastskip% Added to deal with last question exactly filling the page
  \mark{\noexpand\@lastturn}%
  \clearpage
  \if@answers\do@answers\fi}

\pagestyle{exam}

% This was added to deal with problems about initialization order during
% processing of \AtBeginDocument hooks.  Might not be needed now the front
% page isn't done that way.
\def\default@color{Black}

\parindent=0pt
\parskip=\medskipamount
\parsep=0pt
\sloppy

\endinput
