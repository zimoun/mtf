INTERPOLATION=1;

Merge "dir0-aR.pos";
Merge "dir0-cR.pos";

Plugin(MathEval).Expression0= "v0 + w0";
Plugin(MathEval).TimeStep=0;
Plugin(MathEval).View=0;
Plugin(MathEval).OtherTimeStep=0;
Plugin(MathEval).OtherView=1;
Plugin(MathEval).ForceInterpolation=INTERPOLATION;
Plugin(MathEval).PhysicalRegion=-1;
Plugin(MathEval).Run;
View[2].Name = " dir0-r" ;


Save View[2] "dir0-R.pos";

Delete View[2];
Delete View[1];
Delete View[0];


Merge "dir0-aI.pos";
Merge "dir0-cI.pos";

Plugin(MathEval).Expression0= "v0 + w0";
Plugin(MathEval).TimeStep=0;
Plugin(MathEval).View=0;
Plugin(MathEval).OtherTimeStep=0;
Plugin(MathEval).OtherView=1;
Plugin(MathEval).ForceInterpolation=INTERPOLATION;
Plugin(MathEval).PhysicalRegion=-1;
Plugin(MathEval).Run;
View[2].Name = " dir0-i" ;

Save View[2] "dir0-I.pos";

Delete View[2];
Delete View[1];
Delete View[0];




Merge "dir1-aR.pos";
Merge "dir1-bR.pos";

Plugin(MathEval).Expression0= "v0 + w0";
Plugin(MathEval).TimeStep=0;
Plugin(MathEval).View=0;
Plugin(MathEval).OtherTimeStep=0;
Plugin(MathEval).OtherView=1;
Plugin(MathEval).ForceInterpolation=INTERPOLATION;
Plugin(MathEval).PhysicalRegion=-1;
Plugin(MathEval).Run;
View[2].Name = " dir1-r" ;

Save View[2] "dir1-R.pos";

Delete View[2];
Delete View[1];
Delete View[0];


Merge "dir1-aI.pos";
Merge "dir1-bI.pos";

Plugin(MathEval).Expression0= "v0 + w0";
Plugin(MathEval).TimeStep=0;
Plugin(MathEval).View=0;
Plugin(MathEval).OtherTimeStep=0;
Plugin(MathEval).OtherView=1;
Plugin(MathEval).ForceInterpolation=INTERPOLATION;
Plugin(MathEval).PhysicalRegion=-1;
Plugin(MathEval).Run;
View[2].Name = " dir1-i" ;

Save View[2] "dir1-I.pos";

Delete View[2];
Delete View[1];
Delete View[0];


Merge "dir2-cR.pos";
Merge "dir2-bR.pos";

Plugin(MathEval).Expression0= "v0 + w0";
Plugin(MathEval).TimeStep=0;
Plugin(MathEval).View=0;
Plugin(MathEval).OtherTimeStep=0;
Plugin(MathEval).OtherView=1;
Plugin(MathEval).ForceInterpolation=INTERPOLATION;
Plugin(MathEval).PhysicalRegion=-1;
Plugin(MathEval).Run;
View[2].Name = " dir2-r" ;

Save View[2] "dir2-R.pos";

Delete View[2];
Delete View[1];
Delete View[0];


Merge "dir2-cI.pos";
Merge "dir2-bI.pos";

Plugin(MathEval).Expression0= "v0 + w0";
Plugin(MathEval).TimeStep=0;
Plugin(MathEval).View=0;
Plugin(MathEval).OtherTimeStep=0;
Plugin(MathEval).OtherView=1;
Plugin(MathEval).ForceInterpolation=INTERPOLATION;
Plugin(MathEval).PhysicalRegion=-1;
Plugin(MathEval).Run;
View[2].Name = " dir2-I" ;

Save View[2] "dir2-I.pos";

Delete View[2];
Delete View[1];
Delete View[0];
