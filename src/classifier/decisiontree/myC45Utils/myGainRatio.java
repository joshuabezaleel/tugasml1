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
public class myGainRatio {

    public final double splitCritValue(myDistribution bags, double totalnoInst, double numerator) {
        double denumerator;
        double noUnknown;
        double unknownRate;
        int i;
    
        // Compute split info.
        denumerator = splitEnt(bags,totalnoInst);

        // Test if split is trivial.
        if (Utils.eq(denumerator,0))
            return 0;  
        denumerator = denumerator/totalnoInst;

        return numerator/denumerator;
    }

    private double splitEnt(myDistribution bags, double totalnoInst) {
        double returnValue = 0;
        double noUnknown;
        int i;
    
        noUnknown = totalnoInst-bags.total();
        if (Utils.gr(bags.total(),0)) {
            for (i=0; i<bags.numBags(); i++)
                returnValue = returnValue-logFunc(bags.perBag(i));
            returnValue = returnValue-logFunc(noUnknown);
            returnValue = returnValue+logFunc(totalnoInst);
        }
        return returnValue;
    }

    public final double logFunc(double num) {
        double log2 = Math.log(2);
        
        if (num < 1e-6)
            return 0;
        else
            return num*Math.log(num)/log2;
    }
    
}
