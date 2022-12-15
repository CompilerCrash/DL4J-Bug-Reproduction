import org.deeplearning4j.datasets.iterator.RandomDataSetIterator;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.nd4j.autodiff.listeners.impl.ScoreListener;
import org.nd4j.autodiff.listeners.records.History;
import org.nd4j.autodiff.listeners.records.LossCurve;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.weightinit.impl.OneInitScheme;
import org.nd4j.weightinit.impl.XavierInitScheme;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.deeplearning4j.datasets.iterator.RandomDataSetIterator.Values.ONES;
import static org.nd4j.linalg.api.buffer.DataType.FLOAT;

public class Loss {

    public static void main(String[] args) {
        test1();
        System.out.println("=".repeat(100));
        test2();
    }

    public static void test1() {
        int batchSize = 4;
        int modelDim = 8;

        SameDiff sd = SameDiff.create();

        SDVariable features = sd.placeHolder("features", FLOAT, batchSize, modelDim);
        SDVariable labels = sd.placeHolder("labels", FLOAT, batchSize, modelDim);
        SDVariable bias = sd.var("bias", new OneInitScheme('c'), FLOAT, modelDim);
        SDVariable predictions = features.add("predictions", bias);
        sd.loss.meanSquaredError("loss", labels, predictions, null);

        TrainingConfig config = new TrainingConfig.Builder()
                .updater(new Adam(0.1))
                .dataSetFeatureMapping("features")
                .dataSetLabelMapping("labels")
                // .lossVariables(List.of("loss")) // this line fixes the problem
                .build();
        sd.setTrainingConfig(config);

        // Task: output must be equal to input
        DataSetIterator iterator = new RandomDataSetIterator(1, new long[]{batchSize, modelDim}, new long[]{batchSize, modelDim}, ONES, ONES);

        // ScoreListener will consistently report a loss of 0
        History hist = sd.fit(iterator, 10, new ScoreListener(1));

        // The recorded loss curve is also constantly 0
        LossCurve curve = hist.lossCurve();
        System.out.println("Loss curve:");
        System.out.println(curve.getLossNames());
        System.out.println(curve.getLossValues());

        // However, the loss calculated here is > 0
        Map<String, INDArray> map = sd.output(iterator, "loss", "bias");
        System.out.println("Final loss: " + map.get("loss"));
        System.out.println("Final bias: " + map.get("bias"));
    }

    // Source: https://github.com/deeplearning4j/deeplearning4j/issues/9806
    public static void test2() {
        int nIn = 4;
        int nOut = 1;
        int NUM_SAMPLES = 300;
        int epoches = 10;
        int minibatch = 3;

        SameDiff sd = SameDiff.create();

        // First: Let's create our placeholders. Shape: [minibatch, in/out]
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, nIn);
        SDVariable labels = sd.placeHolder("labels", DataType.FLOAT, -1, nOut);

        // Second: let's create our variables
        SDVariable weights = sd.var("weights", new XavierInitScheme('c', nIn, nOut), DataType.FLOAT, nIn, nOut);
        SDVariable bias = sd.var("bias");

        // And define our forward pass:
        SDVariable out = input.mmul(weights).add(bias);     // Note: it's broadcast add here

        // And our loss function
        SDVariable mse = sd.loss.meanSquaredError("mse", labels, out, null);

        // Let's create some mock data for this example:
        Nd4j.getRandom().setSeed(12345);
        INDArray inputArr = Nd4j.rand(minibatch, nIn);
        INDArray labelArr = Nd4j.rand(minibatch, nOut);

        Map<String, INDArray> placeholderData = new HashMap<>();
        placeholderData.put("input", inputArr);
        placeholderData.put("labels", labelArr);

        // Execute forward pass:
        INDArray loss = sd.output(placeholderData, "mse").get("mse");
        System.out.println("MSE: " + loss);

        // Calculate gradients:
        Map<String, INDArray> gradMap = sd.calculateGradients(placeholderData, "weights");
        System.out.println("Weights gradient:");
        System.out.println(gradMap.get("weights"));

        // Mock random dataset for training
        INDArray indFeature = Nd4j.rand(new long[]{NUM_SAMPLES, nIn});
        INDArray indLabel = Nd4j.rand(new long[]{NUM_SAMPLES, nOut});
        DataSet ds = new DataSet(indFeature, indLabel);
        SplitTestAndTrain train_test = ds.splitTestAndTrain(0.7);
        DataSet dsTrain = train_test.getTrain();
        DataSet dsTest = train_test.getTest();
        DataSetIterator trainIter = new ListDataSetIterator<>(Collections.singletonList(dsTrain), minibatch);
        DataSetIterator testIter = new ListDataSetIterator<>(Collections.singletonList(dsTest), minibatch);

        // Train model
        double learningRate = 1e-3;
        TrainingConfig config = new TrainingConfig.Builder()
                .updater(new Sgd(learningRate))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("labels")
                // .lossVariables(Collections.singletonList("mse")) // this line fixes the problem
                .build();
        sd.setTrainingConfig(config);
        sd.setListeners(new ScoreListener(1));
        sd.setLossVariables(mse);
        History hist = sd.fit(trainIter, epoches);
        System.out.println(hist.getLossCurve().getLossValues());
    }
}
