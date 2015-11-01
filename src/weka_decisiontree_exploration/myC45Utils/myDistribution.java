/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka_decisiontree_exploration.myC45Utils;

import java.io.Serializable;
import java.util.Enumeration;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionHandler;
import weka.core.RevisionUtils;
import weka.core.Utils;

/**
 *
 * @author Rakhmatullah Yoga S, Joshua Bezaleel Abednego, Linda Sekawati
 */
public class myDistribution implements Cloneable, Serializable, RevisionHandler {
    private double perClassPerBag[][];
    private double perBag[];
    private double perClass[];
    private double total;

    public myDistribution(Instances instances) {
        perClassPerBag = new double [1][0];
        perBag = new double [1];
        total = 0;
        perClass = new double [instances.numClasses()];
        perClassPerBag[0] = new double [instances.numClasses()];
        Enumeration enu = instances.enumerateInstances();
        while (enu.hasMoreElements())
            add(0,(Instance) enu.nextElement());
    }

    public myDistribution(Instances instances, myClassifierSplitModel modelToUse) throws Exception {
        int index;
        Instance instance;
        double[] weights;

        perClassPerBag = new double [modelToUse.numSubsets()][0];
        perBag = new double [modelToUse.numSubsets()];
        total = 0;
        perClass = new double [instances.numClasses()];
        for (int i = 0; i < modelToUse.numSubsets(); i++)
            perClassPerBag[i] = new double [instances.numClasses()];
        Enumeration enu = instances.enumerateInstances();
        while (enu.hasMoreElements()) {
            instance = (Instance) enu.nextElement();
            index = modelToUse.whichSubset(instance);
            if (index != -1)
                add(index, instance);
            else {
                weights = modelToUse.weights(instance);
                addWeights(instance, weights);
            }
        }
    }

    public myDistribution(myDistribution toMerge) {
        total = toMerge.total;
        perClass = new double [toMerge.numClasses()];
        System.arraycopy(toMerge.perClass, 0, perClass, 0, toMerge.numClasses());
        perClassPerBag = new double [1] [0];
        perClassPerBag[0] = new double [toMerge.numClasses()];
        System.arraycopy(toMerge.perClass, 0, perClassPerBag[0], 0, toMerge.numClasses());
        perBag = new double [1];
        perBag[0] = total;
    }

    public myDistribution(int numBags, int numClasses) {
        int i;

        perClassPerBag = new double [numBags][0];
        perBag = new double [numBags];
        perClass = new double [numClasses];
        for (i=0;i<numBags;i++)
            perClassPerBag[i] = new double [numClasses];
        total = 0;
    }

    public double numIncorrect() {
        return total-numCorrect();
    }

    public int maxBag() {
        double max;
        int maxIndex;
        int i;
    
        max = 0;
        maxIndex = -1;
        for (i=0; i<perBag.length; i++)
            if (Utils.grOrEq(perBag[i],max)) {
                max = perBag[i];
                maxIndex = i;
            }
        return maxIndex;
    }

    public double total() {
        return total;
    }

    public void add(int i, Instance instance) {
        int classIndex;
        double weight;

        classIndex = (int)instance.classValue();
        weight = instance.weight();
        perClassPerBag[i][classIndex] = perClassPerBag[i][classIndex] + weight;
        perBag[i] = perBag[i] + weight;
        perClass[classIndex] = perClass[classIndex] + weight;
        total = total + weight;
    }

    public double numCorrect() {
        return perClass[maxClass()];
    }

    public int maxClass() {
        double maxCount = 0;
        int maxIndex = 0;
        int i;

        for (i=0; i<perClass.length; i++)
            if (Utils.gr(perClass[i],maxCount)) {
                maxCount = perClass[i];
                maxIndex = i;
            }
        return maxIndex;
    }

    public void addWeights(Instance instance, double[] weights) {
        int classIndex;
        int i;

        classIndex = (int)instance.classValue();
        for (i=0; i<perBag.length;i++) {
            double weight = instance.weight() * weights[i];
            perClassPerBag[i][classIndex] = perClassPerBag[i][classIndex] + weight;
            perBag[i] = perBag[i] + weight;
            perClass[classIndex] = perClass[classIndex] + weight;
            total = total + weight;
        }
    }

    public double perClass(int maxClass) {
        return perClass[maxClass];
    }

    public int numClasses() {
        return perClass.length;
    }

    public double perBag(int i) {
        return perBag[i];
    }

    public final boolean check(int m_minNoObj) {
        int counter = 0;
        int i;

        for (i=0;i<perBag.length;i++)
            if (Utils.grOrEq(perBag[i], m_minNoObj))
                counter++;
        return counter > 1;
    }

    public void shiftRange(int from, int to, Instances source, int startIndex, int lastPlusOne) {
        int classIndex;
        double weight;
        Instance instance;
        int i;

        for (i = startIndex; i < lastPlusOne; i++) {
            instance = (Instance) source.instance(i);
            classIndex = (int)instance.classValue();
            weight = instance.weight();
            perClassPerBag[from][classIndex] -= weight;
            perClassPerBag[to][classIndex] += weight;
            perBag[from] -= weight;
            perBag[to] += weight;
        }
    }

    public void addRange(int bagIndex, Instances source, int startIndex, int lastPlusOne) {
        double sumOfWeights = 0;
        int classIndex;
        Instance instance;
        int i;

        for (i = startIndex; i < lastPlusOne; i++) {
            instance = (Instance) source.instance(i);
            classIndex = (int)instance.classValue();
            sumOfWeights = sumOfWeights+instance.weight();
            perClassPerBag[bagIndex][classIndex] += instance.weight();
            perClass[classIndex] += instance.weight();
        }
        perBag[bagIndex] += sumOfWeights;
        total += sumOfWeights;
    }

    public int numBags() {
        return perBag.length;
    }

    public double perClassPerBag(int i, int j) {
        return perClassPerBag[i][j];
    }

    public void addInstWithUnknown(Instances data, int attIndex) {
        double [] probs;
        double weight,newWeight;
        int classIndex;
        Instance instance;
        int j;

        probs = new double [perBag.length];
        for (j=0; j<perBag.length; j++) {
            if (Utils.eq(total, 0)) {
                probs[j] = 1.0 / probs.length;
            } else {
                probs[j] = perBag[j]/total;
            }
        }
        Enumeration enu = data.enumerateInstances();
        while (enu.hasMoreElements()) {
            instance = (Instance) enu.nextElement();
            if (instance.isMissing(attIndex)) {
                classIndex = (int)instance.classValue();
                weight = instance.weight();
                perClass[classIndex] = perClass[classIndex]+weight;
                total = total+weight;
                for (j = 0; j < perBag.length; j++) {
                    newWeight = probs[j]*weight;
                    perClassPerBag[j][classIndex] = perClassPerBag[j][classIndex]+newWeight;
                    perBag[j] = perBag[j]+newWeight;
                }
            }
        }
    }

    @Override
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 1.12 $");
    }

    public double prob(int classIndex) {
        if (!Utils.eq(total, 0)) {
            return perClass[classIndex]/total;
        } else {
            return 0;
        }
    }

    public double prob(int classIndex, int intIndex) {
        if (Utils.gr(perBag[intIndex],0))
            return perClassPerBag[intIndex][classIndex]/perBag[intIndex];
        else
            return prob(classIndex);
    }
    
}
