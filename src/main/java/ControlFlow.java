import org.deeplearning4j.datasets.iterator.RandomDataSetIterator;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.weightinit.impl.OneInitScheme;

import static org.deeplearning4j.datasets.iterator.RandomDataSetIterator.Values.ONES;

public class ControlFlow {

    public static void main(String[] args) {
        ifCond();
        System.out.println("=".repeat(100));
        whileLoop();
    }

    private static void ifCond() {
        int batchSize = 4;
        int modelDim = 8;

        SameDiff sd = SameDiff.create();

        SDVariable features = sd.placeHolder("features", DataType.FLOAT, batchSize, modelDim);
        SDVariable labels = sd.placeHolder("labels", DataType.FLOAT, batchSize, modelDim);
        SDVariable var = sd.var("variable", new OneInitScheme('c'), DataType.FLOAT, batchSize, modelDim);
        SDVariable predictions = sd.ifCond("predictions", null,
                _sd -> features.sum().gt(0),
                _sd -> features.sub(var),
                _sd -> features.add(var));
        sd.loss.meanSquaredError("loss", labels, predictions, null);

        TrainingConfig config = new TrainingConfig.Builder()
                .updater(new Adam(0.1))
                .dataSetFeatureMapping("features")
                .dataSetLabelMapping("labels")
                .build();
        sd.setTrainingConfig(config);

        DataSetIterator iterator = new RandomDataSetIterator(1, new long[]{batchSize, modelDim}, new long[]{batchSize, modelDim}, ONES, ONES);

        System.out.println(sd.output(iterator, "predictions").get("predictions")); // forward pass works

        sd.fit(iterator, 1); // backward pass throws exception
    }

    private static void whileLoop() {
        int batchSize = 4;
        int modelDim = 8;

        SameDiff sd = SameDiff.create();

        SDVariable features = sd.placeHolder("features", DataType.FLOAT, batchSize, modelDim);
        SDVariable labels = sd.placeHolder("labels", DataType.FLOAT, batchSize, modelDim);
        SDVariable var = sd.var("variable", new OneInitScheme('c'), DataType.FLOAT, batchSize, modelDim);
        SDVariable predictions = sd.whileLoop(new String[]{"predictions"}, null,
                new SDVariable[]{features},
                (_sd, inputs) -> inputs[0].sum().gt(0),
                (_sd, inputs) -> new SDVariable[]{inputs[0].sub(var)})[0];
        sd.loss.meanSquaredError("loss", labels, predictions, null);

        TrainingConfig config = new TrainingConfig.Builder()
                .updater(new Adam(0.1))
                .dataSetFeatureMapping("features")
                .dataSetLabelMapping("labels")
                .build();
        sd.setTrainingConfig(config);

        DataSetIterator iterator = new RandomDataSetIterator(1, new long[]{batchSize, modelDim}, new long[]{batchSize, modelDim}, ONES, ONES);

        System.out.println(sd.output(iterator, "predictions").get("predictions")); // forward pass works

        sd.fit(iterator, 1); // backward pass throws exception
    }
}
