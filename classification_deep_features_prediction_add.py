from cytomine import Cytomine
from cytomine.models import Software, SoftwareParameter


def main(argv):
    with Cytomine.connect_from_cli(argv):
        software = Software(name="Classification_Deep_Features_Prediction",
                            service_name="pyxitSuggestedTermJobService",
                            result_name="ValidateAnnotation").save()

        SoftwareParameter("cytomine_project_id", type="Number", id_software=software.id,
                          index=100, set_by_server=True, required=True).save()
        SoftwareParameter("cytomine_software_id", type="Number", id_software=software.id, default_value=software.id,
                          index=200, set_by_server=True, required=True).save()

        # filtering annotations
        SoftwareParameter("cytomine_images_ids", type="List", id_software=software.id, default_value=None, index=500).save()
        SoftwareParameter("cytomine_users_ids", type="List", id_software=software.id, default_value=None, index=600).save()

        # running parameters
        SoftwareParameter("working_directory", type="String", id_software=software.id, default_value="", index=800, required=True).save()
        SoftwareParameter("model_path", type="String", id_software=software.id, default_value="/tmp/model.pkl", index=900, required=True).save()
        SoftwareParameter("n_jobs", type="Number", id_software=software.id, default_value=1, index=1000, required=True).save()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
