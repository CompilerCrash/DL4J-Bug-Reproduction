import org.deeplearning4j.datasets.iterator.RandomDataSetIterator;
import org.deeplearning4j.datasets.iterator.ReconstructionDataSetIterator;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.weightinit.impl.OneInitScheme;
import org.nd4j.weightinit.impl.XavierInitScheme;
import org.nd4j.weightinit.impl.ZeroInitScheme;

import static org.deeplearning4j.datasets.iterator.RandomDataSetIterator.Values.*;
import static org.nd4j.linalg.api.buffer.DataType.FLOAT;

public class Training {

    public static void main(String[] args) {
        noTrainableVariables();
        System.out.println("=".repeat(60));
        noSuccessWithLinear();
    }

    // FIXME
    public static void noTrainableVariables() {
        int batchSize = 4;
        int modelDim = 8;

        SameDiff sd = SameDiff.create();

        SDVariable features = sd.placeHolder("features", FLOAT, batchSize, modelDim);
        SDVariable labels = sd.placeHolder("labels", FLOAT, batchSize, modelDim);
        SDVariable weights = sd.var("weights", new XavierInitScheme('c', modelDim, modelDim), FLOAT, modelDim, modelDim);
        SDVariable bias = sd.var("bias", new ZeroInitScheme('c'), FLOAT, modelDim);
        SDVariable predictions = sd.nn.linear("predictions", features, weights, bias);
        sd.loss.meanSquaredError("loss", labels, predictions, null);

        TrainingConfig config = new TrainingConfig.Builder()
                .updater(new Adam(0.1))
                .dataSetFeatureMapping("features")
                .dataSetLabelMapping("labels")
                .build();
        sd.setTrainingConfig(config);

        DataSetIterator iterator = new RandomDataSetIterator(1, new long[]{batchSize, modelDim}, new long[]{batchSize, modelDim}, INTEGER_0_10, INTEGER_0_10);

        sd.fit(iterator, 10); // throws exception
    }

    // FIXME
    public static void noSuccessWithLinear() {
        int batchSize = 32;
        int modelDim = 10;

        SameDiff sd = SameDiff.create();

        SDVariable features = sd.placeHolder("features", FLOAT, batchSize, modelDim);
        SDVariable labels = sd.placeHolder("labels", FLOAT, batchSize, modelDim);
        SDVariable weights = sd.var("weights", new OneInitScheme('c'), FLOAT, modelDim, modelDim);
        SDVariable bias = sd.zero("bias", modelDim);
        // SDVariable predictions = features.mmul(weights).add("predictions", bias); // <<< variant 1 (works)
        SDVariable predictions = sd.nn.linear("predictions", features, weights, bias); // <<< variant 2 (doesn't work)
        // SDVariable predictions = sd.nn.reluLayer("predictions", features, weights, bias); // <<< variant 3 (doesn't work)
        sd.loss.meanSquaredError("loss", labels, predictions, null);

        TrainingConfig config = new TrainingConfig.Builder()
                .updater(new Adam(0.1))
                .dataSetFeatureMapping("features")
                .dataSetLabelMapping("labels")
                .build();
        sd.setTrainingConfig(config);

        DataSetIterator iterator = new ReconstructionDataSetIterator(new RandomDataSetIterator(100, new long[]{batchSize, modelDim}, new long[]{}, ONE_HOT, ZEROS));

        sd.fit(iterator, 10);

        Evaluation evaluation = new Evaluation();
        sd.evaluate(iterator, "predictions", evaluation);
        System.out.println(evaluation.stats());
    }
}
