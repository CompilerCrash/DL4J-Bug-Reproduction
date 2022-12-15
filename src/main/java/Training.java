import org.deeplearning4j.datasets.iterator.RandomDataSetIterator;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.weightinit.impl.OneInitScheme;
import org.nd4j.weightinit.impl.XavierInitScheme;

import static org.deeplearning4j.datasets.iterator.RandomDataSetIterator.Values.INTEGER_0_10;
import static org.nd4j.linalg.api.buffer.DataType.FLOAT;

public class Training {

    public static void main(String[] args) {
        int batchSize = 4;
        int modelDim = 8;

        SameDiff sd = SameDiff.create();

        SDVariable features = sd.placeHolder("features", FLOAT, batchSize, modelDim);
        SDVariable labels = sd.placeHolder("labels", FLOAT, batchSize, modelDim);
        SDVariable weights = sd.var("weights", new XavierInitScheme('c', modelDim, modelDim), FLOAT, modelDim, modelDim);
        SDVariable bias = sd.var("bias", new OneInitScheme('c'), FLOAT, modelDim);
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
}
