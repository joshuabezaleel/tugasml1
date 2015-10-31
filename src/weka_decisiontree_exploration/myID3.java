/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka_decisiontree_exploration;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.Utils;
import weka.core.Capabilities.Capability;

import java.util.Enumeration;

/**
 *
 * @author Rakhmatullah Yoga S, Joshua Bezaleel Abednego, Linda Sekawati
 */
public class myID3 extends Classifier{
    
    /** Attribute used for splitting. */
    private Attribute m_Attribute;
    
    /** Class value if node is leaf. */
    private double m_ClassValue;

    /** Class distribution if node is leaf. */
    private double[] m_Distribution;

    /** Class attribute of dataset. */
    private Attribute m_ClassAttribute;
  
    /** The node's successors. */ 
    private myID3[] m_Successors;
    
    /** Menghitung maks dari class value*/
    private double maxClass;
    
    private double maxDataClass(Instances data){
        double[] classOfValue = new double[data.numClasses()];
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            classOfValue[(int) inst.classValue()]++;
        }
        return Utils.maxIndex(classOfValue);
    }
    
    private void makeMyID3(Instances data,double maxClass){
        // Check if no instances have reached this node.
        if (data.numInstances() == 0) {
          m_Attribute = null;
          m_ClassValue = maxClass;
          m_Distribution = new double[data.numClasses()];
          return;
        }

        // Menghitung IG atribut dan memilih yg paling besar
        double[] infoGains = new double[data.numAttributes()];
        boolean hasEmpty = false;
        Enumeration attEnum = data.enumerateAttributes();
        while (attEnum.hasMoreElements()) {
            Attribute att = (Attribute) attEnum.nextElement();
            infoGains[att.index()] = InformationGain(data, att); 
        }
        
        m_Attribute = data.attribute(Utils.maxIndex(infoGains));
        
        // Jika info gain = 0, maka daun dapat ditentukan
        // suksesor pohon 
        if (Utils.eq(infoGains[m_Attribute.index()], 0)) {
            m_Attribute = null;
            m_Distribution = new double[data.numClasses()];
            Enumeration instEnum = data.enumerateInstances();
            while (instEnum.hasMoreElements()) {
                Instance inst = (Instance) instEnum.nextElement();
                m_Distribution[(int) inst.classValue()]++;
            }
          
            // menormalkan distribusi
            Utils.normalize(m_Distribution);
            m_ClassValue = Utils.maxIndex(m_Distribution);
            m_ClassAttribute = data.classAttribute();
        } else {
            Instances[] splitData = new Instances[m_Attribute.numValues()];
            for (int j = 0; j < m_Attribute.numValues(); j++) {
                splitData[j] = new Instances(data, data.numInstances());
            }               
        
            Enumeration instEnum = data.enumerateInstances();
            while (instEnum.hasMoreElements()) {
                Instance inst = (Instance) instEnum.nextElement();
                splitData[(int) inst.value(m_Attribute)].add(inst);
            }
        
            for (int i = 0; i < splitData.length; i++) {
                splitData[i].compactify();
            }
            m_Successors = new myID3[m_Attribute.numValues()];
            for (int j = 0; j < m_Attribute.numValues(); j++) {
                m_Successors[j] = new myID3();
                m_Successors[j].makeMyID3(splitData[j],maxClass);
            }
        }
    }
    
    private double InformationGain(Instances data, Attribute att){
        Instances[] attInstances = new Instances[att.numValues()];
        for (int j = 0; j < att.numValues(); j++) {
            attInstances[j] = new Instances(data, data.numInstances());
        }        
        
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            attInstances[(int) inst.value(att)].add(inst);
        }
        
        for (int i = 0; i < attInstances.length; i++) {
            attInstances[i].compactify();
        }
        
        double infoGain = Entropy(data);
        for (int i=0; i< att.numValues(); i++){
            if (attInstances[i].numInstances() > 0) {
            infoGain -= ((double)attInstances[i].numInstances()/
                    (double)data.numInstances())*Entropy(attInstances[i]);
            }
        }
        return infoGain;
    }
    
    private double Entropy (Instances data){
        double [] classCounts = new double[data.numClasses()];
        //Mengisi array 
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            classCounts[(int) inst.classValue()]++;
        }
        
        double entropy= 0;
        for (int j = 0; j < data.numClasses(); j++) {
            if (classCounts[j] > 0) {
                entropy -= classCounts[j] * Utils.log2(classCounts[j]);
            }
        }
        entropy /= (double) data.numInstances();
        return entropy + Utils.log2(data.numInstances());
    }
    
    public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException {
        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("Id3: no missing values, "+ "please.");   
        }
        if (m_Attribute == null) {
            return m_ClassValue;
        } else {
            return m_Successors[(int) instance.value(m_Attribute)].classifyInstance(instance);
        }
    }
   
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        // attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        // class
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);
        // instances
        result.setMinimumNumberInstances(0);
        return result;
    }

    @Override
    public void buildClassifier(Instances data)throws Exception {
        // can classifier handle the data?
        getCapabilities().testWithFail(data);
        maxClass = maxDataClass(data);
        data = new Instances(data);
        makeMyID3(data,maxClass);
    }
}
