/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier.decisiontree.myC45Utils;

import java.util.Enumeration;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author Rakhmatullah Yoga S, Joshua Bezaleel Abednego, Linda Sekawati
 */
public class mySplit extends myClassifierSplitModel {
    private int m_attIndex;
    private int m_minNoObj;
    private int m_complexityIndex;
    private int m_index;
    private double m_sumOfWeights;
    private double m_splitPoint;
    private double m_infoGain;
    private double m_gainRatio;
    private static myGainRatio gainRatioCrit = new myGainRatio();
    private static myInfoGain infoGainCrit = new myInfoGain();

    public mySplit(int i, int minNoObj, double sumOfWeights) {
        // Get index of attribute to split on.
        m_attIndex = i;
        
        // Set minimum number of objects.
        m_minNoObj = minNoObj;

        // Set the sum of the weights
        m_sumOfWeights = sumOfWeights;
    }

    @Override
    public void buildClassifier(Instances trainInstances) throws Exception {
        // Initialize the remaining instance variables.
        nbSubset = 0;
        m_splitPoint = Double.MAX_VALUE;
        m_infoGain = 0;
        m_gainRatio = 0;

        // Different treatment for enumerated and numeric attributes.
        if (trainInstances.attribute(m_attIndex).isNominal()) {
            m_complexityIndex = trainInstances.attribute(m_attIndex).numValues();
            m_index = m_complexityIndex;
            handleEnumeratedAttribute(trainInstances);
        } else {
            m_complexityIndex = 2;
            m_index = 0;
            trainInstances.sort(trainInstances.attribute(m_attIndex));
            handleNumericAttribute(trainInstances);
        }
    }

    @Override
    public int whichSubset(Instance instance) throws Exception {
        if (instance.isMissing(m_attIndex))
            return -1;
        else {
            if (instance.attribute(m_attIndex).isNominal())
                return (int)instance.value(m_attIndex);
            else
                if (Utils.smOrEq(instance.value(m_attIndex),m_splitPoint))
                    return 0;
                else
                    return 1;
        }
    }

    public final double infoGain() {
        return m_infoGain;
    }

    public final double gainRatio() {
        return m_gainRatio;
    }

    public void setSplitPoint(Instances allInstances) {
        double newSplitPoint = -Double.MAX_VALUE;
        double tempValue;
        Instance instance;
    
        if ((allInstances.attribute(m_attIndex).isNumeric()) && (nbSubset > 1)) {
            Enumeration enu = allInstances.enumerateInstances();
            while (enu.hasMoreElements()) {
                instance = (Instance) enu.nextElement();
                if (!instance.isMissing(m_attIndex)) {
                    tempValue = instance.value(m_attIndex);
                    if (Utils.gr(tempValue,newSplitPoint) &&  Utils.smOrEq(tempValue,m_splitPoint))
                        newSplitPoint = tempValue;
                }
            }
            m_splitPoint = newSplitPoint;
        }
    }

    @Override
    public double[] weights(Instance instance) {
        double [] weights;
        int i;
    
        if (instance.isMissing(m_attIndex)) {
            weights = new double [nbSubset];
            for (i=0; i<nbSubset; i++)
                weights [i] = dist.perBag(i)/dist.total();
            return weights;
        } else {
            return null;
        }
    }

    private void handleEnumeratedAttribute(Instances trainInstances) {
        Instance instance;

        dist = new myDistribution(m_complexityIndex, trainInstances.numClasses());
        
        // Only Instances with known values are relevant.
        Enumeration enu = trainInstances.enumerateInstances();
        while (enu.hasMoreElements()) {
            instance = (Instance) enu.nextElement();
            if (!instance.isMissing(m_attIndex))
                dist.add((int)instance.value(m_attIndex),instance);
        }
    
        // Check if minimum number of Instances in at least two
        // subsets.
        if (dist.check(m_minNoObj)) {
            nbSubset = m_complexityIndex;
            m_infoGain = infoGainCrit.splitCritValue(dist,m_sumOfWeights);
            m_gainRatio = gainRatioCrit.splitCritValue(dist,m_sumOfWeights, m_infoGain);
        }
    }

    private void handleNumericAttribute(Instances trainInstances) {
        int firstMiss;
        int next = 1;
        int last = 0;
        int splitIndex = -1;
        double currentInfoGain;
        double defaultEnt;
        double minSplit;
        Instance instance;
        int i;

        // Current attribute is a numeric attribute.
        dist = new myDistribution(2,trainInstances.numClasses());
    
        // Only Instances with known values are relevant.
        Enumeration enu = trainInstances.enumerateInstances();
        i = 0;
        while (enu.hasMoreElements()) {
            instance = (Instance) enu.nextElement();
            if (instance.isMissing(m_attIndex))
                break;
            dist.add(1,instance);
            i++;
        }
        firstMiss = i;
	
        // Compute minimum number of Instances required in each
        // subset.
        minSplit =  0.1 * (dist.total()) / ((double)trainInstances.numClasses());
        if (Utils.smOrEq(minSplit,m_minNoObj)) 
            minSplit = m_minNoObj;
        else
            if (Utils.gr(minSplit,25))
                minSplit = 25;
	
        // Enough Instances with known values?
        if (Utils.sm((double)firstMiss,2*minSplit))
            return;
    
        // Compute values of criteria for all possible split indices.
        defaultEnt = infoGainCrit.oldEnt(dist);
        while (next < firstMiss) {
            if (trainInstances.instance(next-1).value(m_attIndex)+1e-5 < trainInstances.instance(next).value(m_attIndex)) {
                // Move class values for all Instances up to next possible split point.
                dist.shiftRange(1,0,trainInstances,last,next);
	
                // Check if enough Instances in each subset and compute values for criteria.
                if (Utils.grOrEq(dist.perBag(0),minSplit) && Utils.grOrEq(dist.perBag(1),minSplit)) {
                    currentInfoGain = infoGainCrit.splitCritValue(dist,m_sumOfWeights, defaultEnt);
                    if (Utils.gr(currentInfoGain,m_infoGain)) {
                        m_infoGain =  currentInfoGain;
                        splitIndex = next-1;
                    }
                    m_index++;
                }
                last = next;
            }
            next++;
        }
        
        // Was there any useful split?
        if (m_index == 0)
            return;
    
        // Compute modified information gain for best split.
        m_infoGain = m_infoGain-(Utils.log2(m_index)/m_sumOfWeights);
        if (Utils.smOrEq(m_infoGain,0))
            return;
    
        // Set instance variables' values to values for best split.
        nbSubset = 2;
        m_splitPoint = (trainInstances.instance(splitIndex+1).value(m_attIndex)+trainInstances.instance(splitIndex).value(m_attIndex))/2;

        // In case we have a numerical precision problem we need to choose the smaller value
        if (m_splitPoint == trainInstances.instance(splitIndex + 1).value(m_attIndex)) {
            m_splitPoint = trainInstances.instance(splitIndex).value(m_attIndex);
        }

        // Restore distributioN for best split.
        dist = new myDistribution(2,trainInstances.numClasses());
        dist.addRange(0,trainInstances,0,splitIndex+1);
        dist.addRange(1,trainInstances,splitIndex+1,firstMiss);

        // Compute modified gain ratio for best split.
        m_gainRatio = gainRatioCrit.splitCritValue(dist,m_sumOfWeights, m_infoGain);
    }
    
    public final int attIndex() {
        return m_attIndex;
    }
    
}
