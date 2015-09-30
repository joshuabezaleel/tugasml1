/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka_decisiontree_exploration.myC45Utils;

import java.util.Enumeration;
import weka.classifiers.trees.j48.Distribution;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author Rakhmatullah Yoga S
 */
public class myDistribution {
    protected static double log2 = Math.log(2);
    
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
    public myDistribution(Instances data, myC45ClassifierSplitModel splitModel) throws Exception {
        int index;
        Instance instance;
        double[] weights;
        
        perClassPerBag = new double [splitModel.nbSubset()][0];
        perBag = new double [splitModel.nbSubset()];
        total = 0;
        perClass = new double [data.numClasses()];
        for (int i = 0; i < splitModel.nbSubset(); i++)
            perClassPerBag[i] = new double [data.numClasses()];
        Enumeration enu = data.enumerateInstances();
        while (enu.hasMoreElements()) {
            instance = (Instance) enu.nextElement();
            index = splitModel.whichSubset(instance);
            if (index != -1)
                add(index, instance);
            else {
                weights = splitModel.Weights(instance);
                addWeights(instance, weights);
            }
        }
    }

    public myDistribution(int numBags, int numClasses) {
        perClassPerBag = new double [numBags][0];
        perBag = new double [numBags];
        perClass = new double [numClasses];
        for (int i = 0; i < numBags; i++)
            perClassPerBag[i] = new double [numClasses];
        total = 0;
    }
    public final void shiftRange(int from, int to, Instances source, int startIndex, int lastPlusOne) throws Exception {
        int classIndex;
        double weight;
        Instance instance;
        
        for (int i = startIndex; i < lastPlusOne; i++) {
            instance = (Instance) source.instance(i);
            classIndex = (int)instance.classValue();
            weight = instance.weight();
            perClassPerBag[from][classIndex] -= weight;
            perClassPerBag[to][classIndex] += weight;
            perBag[from] -= weight;
            perBag[to] += weight;
        }
    }
    public final void addRange(int bagIndex, Instances source, int startIndex, int lastPlusOne) throws Exception {
        double sumOfWeights = 0;
        int classIndex;
        Instance instance;
        
        for (int i = startIndex; i < lastPlusOne; i++) {
            instance = (Instance) source.instance(i);
            classIndex = (int)instance.classValue();
            sumOfWeights = sumOfWeights+instance.weight();
            perClassPerBag[bagIndex][classIndex] += instance.weight();
            perClass[classIndex] += instance.weight();
        }
        perBag[bagIndex] += sumOfWeights;
        total += sumOfWeights;
    }
    public final void addInstWithUnknown(Instances source, int attIndex) throws Exception {
        double [] probs;
        double weight,newWeight;
        int classIndex;
        Instance instance;
        
        probs = new double [perBag.length];
        for (int j = 0; j < perBag.length;j++) {
            if (Utils.eq(total, 0)) {
                probs[j] = 1.0 / probs.length;
            } else {
                probs[j] = perBag[j]/total;
            }
        }
        Enumeration enu = source.enumerateInstances();
        while (enu.hasMoreElements()) {
            instance = (Instance) enu.nextElement();
            if (instance.isMissing(attIndex)) {
                classIndex = (int)instance.classValue();
                weight = instance.weight();
                perClass[classIndex] = perClass[classIndex]+weight;
                total = total+weight;
                for (int j = 0; j < perBag.length; j++) {
                    newWeight = probs[j]*weight;
                    perClassPerBag[j][classIndex] = perClassPerBag[j][classIndex]+newWeight;
                    perBag[j] = perBag[j]+newWeight;
                }
            }
        }
    }
    public final boolean check(double minNoObj) {
        int counter = 0;
        int i;
        
        for (i=0;i<perBag.length;i++)
            if (Utils.grOrEq(perBag[i],minNoObj))
                counter++;
        if (counter > 1)
            return true;
        else
            return false;
    }
    public final void addWeights(Instance instance, double [] weights) throws Exception {
        int classIndex;
        int i;
        
        classIndex = (int)instance.classValue();
        for (i=0;i<perBag.length;i++) {
            double weight = instance.weight() * weights[i];
            perClassPerBag[i][classIndex] = perClassPerBag[i][classIndex] + weight;
            perBag[i] = perBag[i] + weight;
            perClass[classIndex] = perClass[classIndex] + weight;
            total += weight;
        }
    }
    
    public final void add(int idxBag, Instance instance) {
        int classIndex;
        double weight;
        
        classIndex = (int)instance.classValue();
        weight = instance.weight();
        perClassPerBag[idxBag][classIndex] = perClassPerBag[idxBag][classIndex]+weight;
        perBag[idxBag] = perBag[idxBag]+weight;
        perClass[classIndex] = perClass[classIndex]+weight;
        total = total+weight;
    }
    public final double perBag(int bagIndex) {
        return perBag[bagIndex];
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
    public int maxBag() {
        double max;
        int idxMax;
        
        max = 0;
        idxMax = -1;
        for (int i = 0; i < perBag.length; i++) {
            if(Utils.grOrEq(perBag[i], max)) {
                max = perBag[i];
                idxMax = i;
            }
        }
        return idxMax;
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
    public double numCorrect() {
        return perClass[maxClass()];
    }
    public double numIncorrect() {
        return total - numCorrect();
    }

    public int numClasses() {
        return perClass.length;
    }

    public int numBags() {
        return perBag.length;
    }

    public double perClassPerBag(int bagIndex, int classIndex) {
        return perClassPerBag[bagIndex][classIndex];
    }
    
    //GainRatioSplitCrit
    public static double gainRatioSplitCritValue(myDistribution bags) {

    double numerator;
    double denumerator;
    
    numerator = oldEnt(bags)-newEnt(bags);

    // Splits with no gain are useless.
    if (Utils.eq(numerator,0))
      return Double.MAX_VALUE;
    denumerator = splitEnt(bags);
    
    // Test if split is trivial.
    if (Utils.eq(denumerator,0))
      return Double.MAX_VALUE;
    
    //  We take the reciprocal value because we want to minimize the
    // splitting criterion's value.
    return denumerator/numerator;
  }

  /**
   * This method computes the gain ratio in the same way C4.5 does.
   *
   * @param bags the distribution
   * @param totalnoInst the weight of ALL instances
   * @param numerator the info gain
   */
  public static double gainRatioSplitCritValue(myDistribution bags, double totalnoInst,
				     double numerator){
    
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
  
  /**
   * Help method for computing the split entropy.
   */
  public static double splitEnt(myDistribution bags,double totalnoInst){
    
    double returnValue = 0;
    double noUnknown;
    int i;
    
    noUnknown = totalnoInst-bags.total();
    if (Utils.gr(bags.total(),0)){
      for (i=0;i<bags.numBags();i++)
	returnValue = returnValue-logFunc(bags.perBag(i));
      returnValue = returnValue-logFunc(noUnknown);
      returnValue = returnValue+logFunc(totalnoInst);
    }
    return returnValue;
  }

    public static double oldEnt(myDistribution bags) {
        double returnValue = 0;
        int j;

        for (j=0;j<bags.numClasses();j++)
            returnValue = returnValue+logFunc(bags.perClass(j));
        return logFunc(bags.total())-returnValue; 
    }

    public static double newEnt(myDistribution bags) {
        double returnValue = 0;
    int i,j;

    for (i=0;i<bags.numBags();i++){
      for (j=0;j<bags.numClasses();j++)
	returnValue = returnValue+logFunc(bags.perClassPerBag(i,j));
      returnValue = returnValue-logFunc(bags.perBag(i));
        }
        return -returnValue;
    }

    public static double splitEnt(myDistribution bags) {
        double returnValue = 0;
        int i;

        for (i=0;i<bags.numBags();i++)
          returnValue = returnValue+logFunc(bags.perBag(i));
        return logFunc(bags.total())-returnValue;
    }

    public static double logFunc(double num) {
        
    // Constant hard coded for efficiency reasons
    if (num < 1e-6)
      return 0;
    else
      return num*Math.log(num)/log2;
    }
    
    //InfoGainSplitCrit
  public final double infoGainSplitCritValue(myDistribution bags) {

    double numerator;
        
    numerator = oldEnt(bags)-newEnt(bags);

    // Splits with no gain are useless.
    if (Utils.eq(numerator,0))
      return Double.MAX_VALUE;
        
    // We take the reciprocal value because we want to minimize the
    // splitting criterion's value.
    return bags.total()/numerator;
  }

  public static double infoGainSplitCritValue(myDistribution bags, double totalNoInst) {
    
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


  public static double infoGainSplitCritValue(myDistribution bags,double totalNoInst,
                                     double oldEnt) {
    
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
}
