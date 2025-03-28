def update_parser_with_new_models(parser):
    """
    Update the existing argument parser to include the new model types
    """
    # Update the model-type choices to include new models
    for action in parser._actions:
        if action.dest == 'model_type':
            # Save the existing help text
            help_text = action.help
            # Remove the old action
            parser._actions.remove(action)
            parser._option_string_actions.pop(action.option_strings[0])
            # Add a new action with updated choices
            parser.add_argument('--model-type', type=str, required=True,
                                choices=['cnet', 'rfe', 'lgr', 'edger', 'limma', 
                                         'rsf', 'svm', 'rf', 'gb'],
                                help=help_text)
            break
    
    return parser

# Usage example - to be inserted in the main script
# parser = ArgumentParser()
# ... [original argument definitions]
# parser = update_parser_with_new_models(parser)
# args = parser.parse_args()
