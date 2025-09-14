from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    """Class to store all the settings of the application."""

    OPENAI_API_KEY: str

    # Add your environment variables here and add them to the .env file

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
    )

    @classmethod
    def customise_sources(
        cls,
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customise the settings sources order.

        Order: dotenv, file secrets, environment variables, then initialization.
        """
        return (
            dotenv_settings,
            file_secret_settings,
            env_settings,
            init_settings,
        )


settings = Settings()