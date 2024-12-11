checkpoint_dir=$1

landlord1_path=$landlord1_dir`ls -v "$checkpoint_dir"landlord1_weights* | tail -1`
landlord2_path=$landlord2_dir`ls -v "$checkpoint_dir"landlord2_weights* | tail -1`
landlord3_path=$landlord3_dir`ls -v "$checkpoint_dir"landlord3_weights* | tail -1`
landlord4_path=$landlord4_dir`ls -v "$checkpoint_dir"landlord4_weights* | tail -1`

echo $landlord1_path
echo $landlord2_path
echo $landlord3_path
echo $landlord4_path

mkdir -p most_recent_model

cp $landlord1_path most_recent_model/landlord1.ckpt
cp $landlord2_path most_recent_model/landlord2.ckpt
cp $landlord3_path most_recent_model/landlord3.ckpt
cp $landlord4_path most_recent_model/landlord4.ckpt
