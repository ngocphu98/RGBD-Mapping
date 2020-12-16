#include"printInformation.h"


void printTranform(Transform pose)
{
	UDEBUG("Tranform matrix:");
	UDEBUG("%f %f %f", pose.r11(), pose.r12(), pose.r13());
	UDEBUG("%f %f %f", pose.r21(),  pose.r22(), pose.r23());
	UDEBUG("%f %f %f", pose.r31(), pose.r32(), pose.r33());	
}