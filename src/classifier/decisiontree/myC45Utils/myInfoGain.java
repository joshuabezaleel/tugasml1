/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier.decisiontree.myC45Utils;

import weka.core.Utils;

/**
 *
 * @author Rakhmatullah Yoga S, Joshua Bezaleel Abednego, Linda Sekawati
 */
public class myInfoGain {

    public final double splitCritValue(myDistribution bags, double totalNoInst) {
        double numerator;
        double noUnknown;
        double unknownRate;
        int i;
    
        noUnknown = totalNoInst-bags.total();
        unknownRate = noUnknown/totalNoInst;
        numerator = (oldEnt(bags)-newEnt(bags));
        numerator = (1-unknownRate)*numerator;

        // Splits with no gain are useless.
        if (Utils.eq(numerator,0))
            return 0;

        return numerator/bags.total();
    }

    public double oldEnt(myDistribution bags) {
        double returnValue = 0;
        int j;

        for (j=0;j<bags.numClasses();j++)
            returnValue = returnValue+logFunc(bags.perClass(j));
        return logFunc(bags.total())-returnValue;
    }

    public double splitCritValue(myDistribution bags, double totalNoInst, double oldEnt) {
        double numerator;
        double noUnknown;
        double unknownRate;
        int i;

        noUnknown = totalNoInst-bags.total();
        unknownRate = noUnknown/totalNoInst;
        numerator = (oldEnt-newEnt(bags));
        numerator = (1-unknownRate)*numerator;

        // Splits with no gain are useless.
        if (Utils.eq(numerator,0))
            return 0;

        return numerator/bags.total();
    }

    private double newEnt(myDistribution bags) {
        double returnValue = 0;
        int i,j;

        for (i=0;i<bags.numBags();i++) {
            for (j=0;j<bags.numClasses();j++)
                returnValue = returnValue+logFunc(bags.perClassPerBag(i,j));
            returnValue = returnValue-logFunc(bags.perBag(i));
        }
        return -returnValue;
    }

    private double logFunc(double num) {
        double log2 = Math.log(2);
        
        // Constant hard coded for efficiency reasons
        if (num < 1e-6)
          return 0;
        else
          return num*Math.log(num)/log2;
    }
    
}
