/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka_decisiontree_exploration.myC45Utils;

import java.util.Enumeration;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author Rakhmatullah Yoga S
 */
public class myDistribution {
    
    private double perClassPerBag[][];
    private double perBag[];
    private double perClass[];
    private double total;
    
    public myDistribution(Instances data) {
        perClassPerBag = new double [1][0];
        perBag = new double [1];
        total = 0;
        perClass = new double [data.numClasses()];
        perClassPerBag[0] = new double [data.numClasses()];
        Enumeration enu = data.enumerateInstances();
        while (enu.hasMoreElements())
          add(0,(Instance) enu.nextElement());
    }
    public void add(int idxBag, Instance instance) {
        int classIndex;
        double weight;
        
        classIndex = (int)instance.classValue();
        weight = instance.weight();
        perClassPerBag[idxBag][classIndex] = perClassPerBag[idxBag][classIndex]+weight;
        perBag[idxBag] = perBag[idxBag]+weight;
        perClass[classIndex] = perClass[classIndex]+weight;
        total = total+weight;
    }
    public double perClass(int classIndex) {
        return perClass[classIndex];
    }
    public double total() {
        return total;
    }
    public int maxClass() {
        double maxCount = 0;
        int maxIndex = 0;
        int i;

        for (i=0;i<perClass.length;i++)
          if (Utils.gr(perClass[i],maxCount)) {
            maxCount = perClass[i];
            maxIndex = i;
          }

        return maxIndex;
    }
    public double prob(int idxClass) {
        if(Utils.eq(total, 0))
            return perClass[idxClass]/total;
        else
            return 0;
    }
    public double prob(int idxClass, int idxInt) {
        if(Utils.gr(perBag[idxInt], 0))
            return perClassPerBag[idxInt][idxClass]/perBag[idxInt];
        else
            return prob(idxClass);
    }
}
