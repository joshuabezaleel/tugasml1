/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka_decisiontree_exploration.myC45Utils;

/**
 *
 * @author Rakhmatullah Yoga S
 */
public class myC45SplitCriteria {
    public final double oldEnt(myDistribution bags) {
        double returnValue = 0;
        int j;
        
        for (j=0;j<bags.numClasses();j++)
            returnValue = returnValue+logFunc(bags.perClass(j));
        return logFunc(bags.total())-returnValue; 
    }
    public final double newEnt(myDistribution bags) {
        double returnValue = 0;
        int i,j;
        
        for (i=0;i<bags.numBags();i++){
            for (j=0;j<bags.numClasses();j++)
                returnValue = returnValue+logFunc(bags.perClassPerBag(i,j));
            returnValue = returnValue-logFunc(bags.perBag(i));
        }
        return -returnValue;
    }
    public double logFunc(double num) {
        if (num < 1e-6)
            return 0;
        else
            return num*Math.log(num)/Math.log(2);
    }
}
