
//
// Add = View[0] + View[1]
//

INTERPOLATION=1;


Plugin(MathEval).Expression0= "v0 + w0";
Plugin(MathEval).TimeStep=0;
Plugin(MathEval).View=0;
Plugin(MathEval).OtherTimeStep=0;
Plugin(MathEval).OtherView=1;
Plugin(MathEval).ForceInterpolation=INTERPOLATION;
Plugin(MathEval).PhysicalRegion=-1;
Plugin(MathEval).Run;
View[2].Name = " add" ;

Save View[2] "add.pos";
